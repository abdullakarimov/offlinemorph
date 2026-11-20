package com.offlinemorph.android.feature.ancestry

import android.graphics.Bitmap
import android.net.Uri

/** State of the ancestry blend pipeline. */
sealed interface AncestryUiState {
    data object Idle : AncestryUiState
    data class Loading(val message: String) : AncestryUiState
    data class Success(val resultBitmap: Bitmap, val appliedRatio: Float) : AncestryUiState
    data class Error(val message: String) : AncestryUiState
    /** Feature is disabled by the build-time flag. */
    data object Unavailable : AncestryUiState
    /** Feature is enabled but a required model file is not yet on-device. */
    data object ModelNotReady : AncestryUiState
}

/** Full-screen state for the ancestry screen. */
data class AncestryScreenState(
    val faceAUri: Uri? = null,
    val faceBUri: Uri? = null,
    /** Base portrait the blended identity is rendered onto; null = use face A's photo. */
    val baseUri: Uri? = null,
    /** Weight of face A in [0, 1]. */
    val blendRatio: Float = 0.5f,
    val intensity: Float = 1.0f,
    val ancestryState: AncestryUiState = AncestryUiState.Idle,
    val isWorking: Boolean = false,
) {
    val canRun: Boolean get() = faceAUri != null && faceBUri != null && !isWorking
}
