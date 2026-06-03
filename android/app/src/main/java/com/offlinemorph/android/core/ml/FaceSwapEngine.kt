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
