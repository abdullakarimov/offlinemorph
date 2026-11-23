package com.offlinemorph.android.core.ml

import android.graphics.Bitmap

data class SwapRunResult(
    val statusMessage: String,
    val outputBitmap: Bitmap? = null,
)

data class SwapRequest(
    val sourceBitmap: Bitmap,
    val targetBitmap: Bitmap,
    val enhancerEnabled: Boolean = false,
    /** Index into [FaceAnalysisSummary.allDetectedFaces] for the face to replace when [faceFilterMode] is [FaceFilterMode.SPECIFIC]. */
    val targetFaceIndex: Int = 0,
    /** Determines which target faces are swapped. */
    val faceFilterMode: FaceFilterMode = FaceFilterMode.SPECIFIC,
    /**
     * Longest-edge cap (px) applied to both bitmaps before inference.
     * Derived from [com.offlinemorph.android.feature.device.ExecutionPolicyManager].
     * 0 = no cap (use full resolution).
     */
    val maxImageSizePx: Int = 0,
    /**
     * InSwapper "pixel boost" factor. With N > 1 the face is aligned at 128·N px and swapped as
     * N² interleaved 128 px sub-crops, yielding an N× sharper face at N² the inference cost.
     */
    val pixelBoost: Int = 1,
    /** Poisson-blend the swapped face into the target so lighting and skin tone match. */
    val seamlessBlend: Boolean = true,
    /**
     * Limit the paste to the target's parsed face region (needs face_parsing.onnx), so hair,
     * hands or glasses in front of the face are kept. Ignored when the model is missing.
     */
    val occlusionAware: Boolean = false,
    /**
     * Optional post-processing of the selected target keypoints before swapping, e.g. temporal
     * stabilisation across video frames. Receives and returns 10-float 5-point arrays.
     */
    val keypointTransform: ((List<FloatArray>) -> List<FloatArray>)? = null,
) {
    /**
     * Returns a copy of this request with both bitmaps downscaled so that their longest
     * edge does not exceed [maxPx]. Returns the same request unchanged if both bitmaps
     * are already within the limit or [maxPx] is 0.
     */
    fun scaledTo(maxPx: Int): SwapRequest {
        if (maxPx <= 0) return this
        return copy(
            sourceBitmap = sourceBitmap.scaledToMax(maxPx),
            targetBitmap = targetBitmap.scaledToMax(maxPx),
            maxImageSizePx = 0,
        )
    }

    private fun Bitmap.scaledToMax(maxPx: Int): Bitmap {
        val longest = maxOf(width, height)
        if (longest <= maxPx) return this
        val scale = maxPx.toFloat() / longest
        return Bitmap.createScaledBitmap(
            this,
            (width * scale).toInt().coerceAtLeast(1),
            (height * scale).toInt().coerceAtLeast(1),
            /* filter= */ true,
        )
    }
}

interface FaceSwapEngine {
    suspend fun runSwap(request: SwapRequest, onProgress: (String) -> Unit = {}): SwapRunResult
}
