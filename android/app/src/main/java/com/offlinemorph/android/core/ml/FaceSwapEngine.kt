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
)

interface FaceSwapEngine {
    suspend fun runSwap(request: SwapRequest, onProgress: (String) -> Unit = {}): SwapRunResult
}
