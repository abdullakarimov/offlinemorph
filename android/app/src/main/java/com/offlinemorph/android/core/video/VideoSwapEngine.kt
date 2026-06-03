package com.offlinemorph.android.core.video

import android.graphics.Bitmap
import android.net.Uri
import java.io.File

data class VideoSwapRequest(
    /** Source face image (same role as in image swap). */
    val sourceBitmap: Bitmap,
    /** URI of the target video (MP4 / MOV). */
    val targetVideoUri: Uri,
    /** When true the GFPGAN enhancer is applied per-frame (slower). */
    val enhancerEnabled: Boolean = false,
    /** Index of the face to replace when [faceFilterMode] is [FaceFilterMode.SPECIFIC]. */
    val targetFaceIndex: Int = 0,
    /** Determines which faces in each frame are swapped. */
    val faceFilterMode: com.offlinemorph.android.core.ml.FaceFilterMode = com.offlinemorph.android.core.ml.FaceFilterMode.SPECIFIC,
)

data class VideoSwapResult(
    /** Path to the output MP4 file, or null when the swap failed. */
    val outputFile: File?,
    val statusMessage: String,
    val framesProcessed: Int = 0,
)

interface VideoSwapEngine {
    /**
     * Processes [request.targetVideoUri] frame-by-frame, replacing the selected face with
     * [request.sourceBitmap] in each frame, and writes the result to a new MP4 file.
     *
     * [onProgress] is called after each frame with (framesProcessed, estimatedTotal, milestone).
     */
    suspend fun swapVideo(
        request: VideoSwapRequest,
        cancellationCheck: () -> Boolean = { false },
        onProgress: (current: Int, total: Int, milestone: String) -> Unit = { _, _, _ -> },
    ): VideoSwapResult
}
