package com.offlinemorph.android.core.ml

import android.graphics.Bitmap
import android.util.Log
import com.offlinemorph.android.feature.models.ModelCatalog
import java.io.File

/**
 * Optional post-swap face restoration with GFPGAN v1.4.
 *
 * GFPGAN was trained on FFHQ-aligned faces, so the face is cropped from the already-swapped
 * image with the [FaceTemplate.FFHQ_512] template (not the tighter inswapper crop), restored,
 * and pasted back with a feathered matte.
 *
 * ONNX contract: input `[1, 3, 512, 512]` RGB in [-1, 1] → output of the same shape.
 */
class OnDeviceFaceEnhancer(
    private val modelsDirectory: File,
    private val sessionFactory: OrtSessionFactory,
) {
    companion object {
        const val MODEL_FILE = ModelCatalog.GFPGAN
        private const val SIZE = 512
        /** Share of the restored face kept in the output (FaceFusion's default blend is 80 %). */
        private const val BLEND = 0.8f
        private const val TAG = "FaceEnhancer"
    }

    fun isAvailable(): Boolean = File(modelsDirectory, MODEL_FILE).isFile

    /**
     * Restores the face described by [fiveKeypoints] in [image].
     *
     * @return a new bitmap, or `null` when the model is missing or inference fails.
     */
    fun enhanceFace(image: Bitmap, fiveKeypoints: FloatArray): Bitmap? {
        if (!isAvailable()) return null
        return runCatching {
            val session = sessionFactory.session(File(modelsDirectory, MODEL_FILE))
            val toCrop = FaceAlignmentOps.alignmentMatrix(fiveKeypoints, FaceTemplate.FFHQ_512, SIZE)
            val crop = FaceAlignmentOps.warpToCrop(image, toCrop, SIZE)
            val input = FaceAlignmentOps.rgbMatToChw(crop, mean = 127.5f, std = 127.5f)
            crop.release()

            val plane = SIZE * SIZE

            val restored = sessionFactory.createFloatTensor(input, longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong()))
                .use { tensor ->
                    val inputName = session.inputNames.firstOrNull() ?: "input"
                    session.run(mapOf(inputName to tensor)).use { result ->
                        OrtValueUtils.extractFloatOutputs(result).firstOrNull { it.data.size == plane * 3 }?.data
                    }
                } ?: return null

            for (i in restored.indices) {
                restored[i] = ((restored[i].coerceIn(-1f, 1f) + 1f) * 0.5f)
            }
            val patch = FaceAlignmentOps.chwToRgbMat(restored, SIZE)
            try {
                FaceAlignmentOps.pasteBack(
                    target = image,
                    patch = patch,
                    patchToTarget = FaceAlignmentOps.invert(toCrop),
                    seamless = false,
                    strength = BLEND,
                )
            } finally {
                patch.release()
            }
        }.getOrElse { e ->
            Log.e(TAG, "Enhancement failed", e)
            null
        }
    }
}
