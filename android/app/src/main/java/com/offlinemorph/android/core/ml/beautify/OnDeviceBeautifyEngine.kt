package com.offlinemorph.android.core.ml.beautify

import android.graphics.Bitmap
import android.util.Log
import com.offlinemorph.android.core.ml.EngineError
import com.offlinemorph.android.core.ml.EngineProgress
import com.offlinemorph.android.core.ml.EngineResult
import com.offlinemorph.android.core.ml.FaceAnalyzer
import com.offlinemorph.android.core.ml.face.FaceClass
import com.offlinemorph.android.core.ml.face.FaceParse
import com.offlinemorph.android.core.ml.face.FaceRegionParser
import com.offlinemorph.android.core.ml.face.FaceWarp
import com.offlinemorph.android.core.ml.face.RegionEffects
import com.offlinemorph.android.core.ml.face.RoiIo
import com.offlinemorph.android.core.ml.face.eroded
import com.offlinemorph.android.core.ml.face.faceKeypoints
import com.offlinemorph.android.feature.models.ModelCatalog
import java.io.File
import kotlin.math.hypot
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

/**
 * On-device portrait beautification.
 *
 * Each detected face (up to [MAX_FACES]) is parsed into regions — with face_parsing.onnx when
 * installed, else from the 5-point landmarks — and edited at the photo's full resolution:
 *
 * - **Skin smoothing**: frequency-separation retouch limited to skin and nose, so eyes, brows,
 *   lips and hair stay sharp.
 * - **Teeth whitening**: inside the inner-mouth region, only tooth-like (bright, low-chroma)
 *   pixels lose their yellow cast and gain a little lightness.
 * - **Eye enlarging**: capped local magnification around each eye.
 * - **Face slimming**: a smooth, capped horizontal compression of the cheeks and jaw toward
 *   the face's centre line.
 *
 * Without the face detector, only skin smoothing is available and it falls back to a gentle
 * whole-image edge-preserving filter.
 */
class OnDeviceBeautifyEngine(
    private val modelsDirectory: File,
    private val faceAnalyzer: FaceAnalyzer,
    private val parser: FaceRegionParser = FaceRegionParser(modelsDirectory),
) : BeautifyEngine {

    override suspend fun runBeautify(
        request: BeautifyRequest,
        onProgress: (EngineProgress) -> Unit,
    ): EngineResult<BeautifyResult> {
        val src = request.sourceBitmap
        val skin = request.skinSmoothing.coerceIn(0f, 1f)
        val eyes = request.eyeEnlarge.coerceIn(0f, 1f)
        val slim = request.faceSlim.coerceIn(0f, 1f)
        val teeth = request.teethWhiten.coerceIn(0f, 1f)
        if (skin + eyes + slim + teeth <= 0f) {
            return EngineResult.Success(BeautifyResult(src.copy(Bitmap.Config.ARGB_8888, false)))
        }
        val faceEditsRequested = eyes + slim + teeth > 0f

        return try {
            if (!File(modelsDirectory, ModelCatalog.DETECTOR).isFile) {
                if (faceEditsRequested) {
                    return EngineResult.Failure(
                        EngineError.ModelNotFound(ModelCatalog.DETECTOR),
                        "Face detection model is not installed. Open Setup → Download Required Models.",
                    )
                }
                onProgress(EngineProgress("Smoothing…", 0.2f))
                return EngineResult.Success(BeautifyResult(globalSmoothing(src, skin)))
            }

            onProgress(EngineProgress("Detecting faces…", 0.05f))
            val faces = faceAnalyzer.analyze(src).faceKeypoints(MAX_FACES)
            if (faces.isEmpty()) {
                return EngineResult.Failure(EngineError.NoFaceDetected("beautify"), "No face found in the photo.")
            }

            val out = src.copy(Bitmap.Config.ARGB_8888, true)
            for ((index, kps) in faces.withIndex()) {
                onProgress(EngineProgress(
                    if (faces.size > 1) "Retouching face ${index + 1} / ${faces.size}…" else "Retouching…",
                    0.15f + 0.8f * index / faces.size,
                ))
                val parse = parser.parse(out, kps)
                beautifyFace(out, parse, kps, skin, eyes, slim, teeth)
            }
            onProgress(EngineProgress("Done", 1f))
            EngineResult.Success(BeautifyResult(out))
        } catch (e: Exception) {
            Log.e(TAG, "Beautify failed", e)
            EngineResult.Failure(EngineError.InferenceFailure(e), e.message ?: "Beautify processing failed.")
        }
    }

    private fun beautifyFace(
        image: Bitmap,
        parse: FaceParse,
        kps: FloatArray,
        skin: Float,
        eyes: Float,
        slim: Float,
        teeth: Float,
    ) {
        val roi = parse.roi ?: return
        var rgb = RoiIo.read(image, roi)
        val iod = hypot(kps[2] - kps[0], kps[3] - kps[1])
        val cropPx = (parse.eyeDistance / 60f).coerceAtLeast(1f) // ~1 px per 60th of the eye distance

        if (skin > 0f) {
            val region = parse.classMask(FaceClass.SKIN, FaceClass.NOSE)
            val inner = region.eroded((cropPx * 2).toInt().coerceAtLeast(1) * 2 + 1)
            region.release()
            parse.project(inner, featherPx = (cropPx * 4).toInt())?.let { mask ->
                RegionEffects.smoothSkin(rgb, mask, skin, iod)
                mask.release()
            }
            inner.release()
        }

        if (teeth > 0f) {
            val mouth = parse.classMask(FaceClass.MOUTH)
            parse.project(mouth, featherPx = cropPx.toInt())?.let { mask ->
                RegionEffects.whitenTeeth(rgb, mask, teeth)
                mask.release()
            }
            mouth.release()
        }

        if (eyes > 0f || slim > 0f) {
            val warp = FaceWarp(rgb.cols(), rgb.rows())
            val ox = roi.left.toFloat(); val oy = roi.top.toFloat()
            if (eyes > 0f) {
                val amount = eyes * FaceWarp.MAX_BULGE * 0.8f
                warp.bulge(kps[0] - ox, kps[1] - oy, iod * 0.42f, amount)
                warp.bulge(kps[2] - ox, kps[3] - oy, iod * 0.42f, amount)
            }
            if (slim > 0f) addSlimField(warp, parse, slim)
            if (!warp.isEmpty) {
                val warped = warp.apply(rgb)
                rgb.release()
                rgb = warped
            }
        }

        RoiIo.write(image, roi, rgb)
        rgb.release()
    }

    /**
     * Face slimming as one smooth horizontal compression in face space (so head tilt doesn't
     * matter). With u = (crop x − centre) / half-width, content is pulled toward the centre line
     * by A·p(u), p(u) = u·exp(−(|u| / 1.1)⁴): linear near the centre (the nose and mouth keep
     * their shape, just narrower), peaking around the cheek line and fading to zero past the
     * face edge. A vertical window limits it to cheeks → jaw → chin. A ≤ 9 % of the half-width
     * keeps the displacement gradient far below 1, so the warp never folds.
     */
    private fun addSlimField(warp: FaceWarp, parse: FaceParse, slim: Float) {
        val half = 0.26f * parse.zoom * parse.size
        val centre = parse.size / 2f
        val amount = slim * 0.09f * half
        fun ty(n: Float) = ((n - 0.5f) * parse.zoom + 0.5f) * parse.size
        val v0 = ty(0.50f); val v1 = ty(0.64f); val v2 = ty(0.88f); val v3 = ty(1.0f)
        val toCrop = parse.roiToCropAffine()
        // Linear part of crop → image, to express a crop-space x offset in image pixels.
        val m = FloatArray(9).also(parse.cropToImage::getValues)
        val ix = m[android.graphics.Matrix.MSCALE_X]
        val iy = m[android.graphics.Matrix.MSKEW_Y]
        warp.addField { x, y, out ->
            val cx = toCrop[0] * x + toCrop[1] * y + toCrop[2]
            val cy = toCrop[3] * x + toCrop[4] * y + toCrop[5]
            if (cy <= v0 || cy >= v3) return@addField
            val w = smoothstep(v0, v1, cy) * (1f - smoothstep(v2, v3, cy))
            val u = (cx - centre) / half
            val au = kotlin.math.abs(u) / 1.1f
            if (au > 2f) return@addField
            val shift = amount * w * u * kotlin.math.exp(-(au * au * au * au))
            out[0] = shift * ix
            out[1] = shift * iy
        }
    }

    private fun smoothstep(e0: Float, e1: Float, x: Float): Float {
        val t = ((x - e0) / (e1 - e0)).coerceIn(0f, 1f)
        return t * t * (3f - 2f * t)
    }

    /** Detector-free fallback: gentle whole-image edge-preserving smoothing. */
    private fun globalSmoothing(src: Bitmap, strength: Float): Bitmap {
        val maxPx = 1024
        val scale = minOf(1f, maxPx.toFloat() / maxOf(src.width, src.height))
        val workW = (src.width * scale).toInt().coerceAtLeast(1)
        val workH = (src.height * scale).toInt().coerceAtLeast(1)
        val work = if (scale < 1f) Bitmap.createScaledBitmap(src, workW, workH, true) else src

        val rgba = Mat()
        Utils.bitmapToMat(work, rgba)
        if (work !== src) work.recycle()
        val rgb = Mat()
        Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)
        rgba.release()

        val sigma = (strength * 60.0).coerceAtLeast(5.0)
        val filtered = Mat()
        Imgproc.bilateralFilter(rgb, filtered, -1, sigma, sigma)
        val blended = Mat()
        Core.addWeighted(filtered, strength.toDouble(), rgb, 1.0 - strength, 0.0, blended)
        rgb.release(); filtered.release()

        val outRgba = Mat()
        Imgproc.cvtColor(blended, outRgba, Imgproc.COLOR_RGB2RGBA)
        blended.release()
        val result = Bitmap.createBitmap(workW, workH, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(outRgba, result)
        outRgba.release()
        if (workW == src.width && workH == src.height) return result
        return Bitmap.createScaledBitmap(result, src.width, src.height, true).also {
            if (it !== result) result.recycle()
        }
    }

    private companion object {
        const val TAG = "BeautifyEngine"
        const val MAX_FACES = 4
    }
}
