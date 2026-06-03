package com.offlinemorph.android.core.ml.hairmakeup

import android.graphics.Bitmap
import android.graphics.Color
import com.offlinemorph.android.core.ml.BitmapTensorConverter
import com.offlinemorph.android.core.ml.EngineError
import com.offlinemorph.android.core.ml.EngineProgress
import com.offlinemorph.android.core.ml.EngineResult
import com.offlinemorph.android.core.ml.OrtSessionFactory
import com.offlinemorph.android.core.ml.OrtValueUtils
import com.offlinemorph.android.feature.models.ModelCatalog
import java.io.File

/**
 * On-device hair & makeup engine backed by a BiSeNet face-parsing ONNX model.
 *
 * The model segments a portrait into 19 semantic regions. We extract hair, lip, and
 * eye-shadow masks from the segmentation and blend the requested colours over them.
 *
 * Model file: [ModelCatalog.FACE_PARSING] ("face_parsing.onnx")
 * Expected input:  (1, 3, 512, 512) float32, ImageNet-normalised RGB.
 * Expected output: (1, 19, 512, 512) float32 class logits.
 */
class OnDeviceHairMakeupEngine(
    private val modelsDirectory: File,
    private val sessionFactory: OrtSessionFactory = OrtSessionFactory(),
) : HairMakeupEngine {

    companion object {
        private const val MODEL_SIZE = 512
        private const val NUM_CLASSES = 19

        // BiSeNet face-parsing class indices (verified by inference on this model):
        // 0=bg, 1=skin, 2=l_brow, 3=r_brow, 4=l_eye, 5=r_eye, 7=l_ear,
        // 10=mouth, 12=u_lip, 13=l_lip, 17=hair
        private const val CLASS_L_BROW = 2
        private const val CLASS_R_BROW = 3
        private const val CLASS_L_EYE = 4
        private const val CLASS_R_EYE = 5
        private const val CLASS_UPPER_LIP = 12
        private const val CLASS_LOWER_LIP = 13
        private const val CLASS_HAIR = 17

        private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    private val tensorConverter = BitmapTensorConverter()

    override suspend fun runHairMakeup(
        request: HairMakeupRequest,
        onProgress: (EngineProgress) -> Unit,
    ): EngineResult<HairMakeupResult> {
        val modelFile = File(modelsDirectory, ModelCatalog.FACE_PARSING)
        if (!modelFile.isFile) {
            return EngineResult.Failure(
                error = EngineError.ModelNotFound(ModelCatalog.FACE_PARSING),
                statusMessage = "Face parsing model is not downloaded. Go to Setup → Download All to get it.",
            )
        }
        return try {
            onProgress(EngineProgress("Running face segmentation…", 0.1f))
            val classMap = runSegmentation(modelFile, request.sourceBitmap)
            onProgress(EngineProgress("Applying colour…", 0.75f))

            var output = request.sourceBitmap.copy(Bitmap.Config.ARGB_8888, true)

            if (request.hairColorHex != null) {
                output = applyColorOverlay(
                    bitmap = output,
                    classMap = classMap,
                    targetClasses = intArrayOf(CLASS_HAIR),
                    hexColor = request.hairColorHex,
                    alpha = request.intensity * 0.85f,
                )
            }
            if (request.lipColorHex != null) {
                output = applyColorOverlay(
                    bitmap = output,
                    classMap = classMap,
                    targetClasses = intArrayOf(CLASS_UPPER_LIP, CLASS_LOWER_LIP),
                    hexColor = request.lipColorHex,
                    alpha = request.intensity * 0.90f,
                )
            }
            if (request.eyeshadowColorHex != null) {
                output = applyColorOverlay(
                    bitmap = output,
                    classMap = classMap,
                    targetClasses = intArrayOf(CLASS_L_BROW, CLASS_R_BROW, CLASS_L_EYE, CLASS_R_EYE),
                    hexColor = request.eyeshadowColorHex,
                    alpha = request.intensity * 0.50f,
                )
            }

            onProgress(EngineProgress("Done", 1f))
            EngineResult.Success(HairMakeupResult(output))
        } catch (e: Exception) {
            EngineResult.Failure(
                error = EngineError.InferenceFailure(e),
                statusMessage = e.message ?: "Hair & makeup inference failed.",
            )
        }
    }

    // ── Segmentation ─────────────────────────────────────────────────────────

    private fun runSegmentation(modelFile: File, bitmap: Bitmap): IntArray {
        val scaled = Bitmap.createScaledBitmap(bitmap, MODEL_SIZE, MODEL_SIZE, true)
        val tensor = tensorConverter.toNormalizedChwFloatArray(
            bitmap = scaled,
            mean = IMAGENET_MEAN,
            std = IMAGENET_STD,
        )
        scaled.recycle()

        sessionFactory.createSession(modelFile).use { session ->
            val inputName = session.inputNames.first()
            sessionFactory.createFloatTensor(
                data = tensor,
                shape = longArrayOf(1L, 3L, MODEL_SIZE.toLong(), MODEL_SIZE.toLong()),
            ).use { inputTensor ->
                session.run(mapOf(inputName to inputTensor)).use { result ->
                    val outputs = OrtValueUtils.extractFloatOutputs(result)
                    check(outputs.isNotEmpty()) { "Face parsing model returned no outputs." }
                    val logits = outputs.first()
                    return argmaxMap(
                        data = logits.data,
                        shape = logits.shape,
                        dstW = bitmap.width,
                        dstH = bitmap.height,
                    )
                }
            }
        }
    }

    /**
     * Argmax over the class axis of a (1, C, H, W) float tensor, then nearest-neighbour
     * upscale to [dstW] × [dstH]. Returns an IntArray indexed [y * dstW + x] → class id.
     */
    private fun argmaxMap(data: FloatArray, shape: LongArray?, dstW: Int, dstH: Int): IntArray {
        val h = shape?.getOrNull(2)?.toInt() ?: MODEL_SIZE
        val w = shape?.getOrNull(3)?.toInt() ?: MODEL_SIZE
        val numC = shape?.getOrNull(1)?.toInt() ?: NUM_CLASSES
        val stride = h * w

        val classMap = IntArray(h * w)
        for (i in 0 until h * w) {
            var maxVal = Float.NEGATIVE_INFINITY
            var maxClass = 0
            for (c in 0 until numC) {
                val v = data[c * stride + i]
                if (v > maxVal) {
                    maxVal = v
                    maxClass = c
                }
            }
            classMap[i] = maxClass
        }

        if (dstW == w && dstH == h) return classMap

        // Nearest-neighbour upscale.
        val scaled = IntArray(dstW * dstH)
        for (dy in 0 until dstH) {
            val sy = dy * h / dstH
            for (dx in 0 until dstW) {
                scaled[dy * dstW + dx] = classMap[sy * w + (dx * w / dstW)]
            }
        }
        return scaled
    }

    // ── Colour overlay ────────────────────────────────────────────────────────

    private fun applyColorOverlay(
        bitmap: Bitmap,
        classMap: IntArray,
        targetClasses: IntArray,
        hexColor: String,
        alpha: Float,
    ): Bitmap {
        val targetColor = Color.parseColor("#$hexColor")
        val tr = Color.red(targetColor) / 255f
        val tg = Color.green(targetColor) / 255f
        val tb = Color.blue(targetColor) / 255f
        // Luminance of the target colour (Rec.601 coefficients).
        val targetLum = 0.299f * tr + 0.587f * tg + 0.114f * tb
        val clampedAlpha = alpha.coerceIn(0f, 1f)

        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        val classSet = targetClasses.toHashSet()
        for (i in pixels.indices) {
            if (classMap[i] !in classSet) continue
            val sr = Color.red(pixels[i]) / 255f
            val sg = Color.green(pixels[i]) / 255f
            val sb = Color.blue(pixels[i]) / 255f

            // Scale the target colour so it has the same luminance as the source pixel.
            // This preserves hair/lip texture and shadows instead of painting bright patches.
            val srcLum = 0.299f * sr + 0.587f * sg + 0.114f * sb
            val lumRatio = if (targetLum > 0.001f) srcLum / targetLum else srcLum
            val colorizedR = (tr * lumRatio).coerceIn(0f, 1f)
            val colorizedG = (tg * lumRatio).coerceIn(0f, 1f)
            val colorizedB = (tb * lumRatio).coerceIn(0f, 1f)

            pixels[i] = Color.argb(
                Color.alpha(pixels[i]),
                ((sr + (colorizedR - sr) * clampedAlpha) * 255f).toInt().coerceIn(0, 255),
                ((sg + (colorizedG - sg) * clampedAlpha) * 255f).toInt().coerceIn(0, 255),
                ((sb + (colorizedB - sb) * clampedAlpha) * 255f).toInt().coerceIn(0, 255),
            )
        }

        val out = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        out.setPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        return out
    }
}
