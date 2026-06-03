package com.offlinemorph.android.feature.videoswap

import android.graphics.Bitmap
import android.net.Uri
import java.io.File

sealed interface VideoSwapUiState {
    data object Idle : VideoSwapUiState
    data class Loading(
        val currentFrame: Int = 0,
        val totalFrames: Int = 0,
        val statusMessage: String = "Processing…",
    ) : VideoSwapUiState
    data class Success(val outputFile: File) : VideoSwapUiState
    data class Error(val message: String) : VideoSwapUiState
}

data class VideoSwapScreenState(
    val sourceUri: Uri? = null,
    val sourceBitmap: Bitmap? = null,
    val targetVideoUri: Uri? = null,
    val enhancerEnabled: Boolean = false,
    val swapState: VideoSwapUiState = VideoSwapUiState.Idle,
    /** Which faces in each frame to swap. */
    val faceFilterMode: com.offlinemorph.android.core.ml.FaceFilterMode = com.offlinemorph.android.core.ml.FaceFilterMode.SPECIFIC,
    /** True when genderage.onnx is installed — enables Male/Female filter modes. */
    val isGenderModelAvailable: Boolean = false,
) {
    val canSwap: Boolean
        get() = sourceBitmap != null && targetVideoUri != null &&
                swapState !is VideoSwapUiState.Loading

    val isWorking: Boolean
        get() = swapState is VideoSwapUiState.Loading
}
