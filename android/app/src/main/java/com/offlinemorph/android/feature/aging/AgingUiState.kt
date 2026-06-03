package com.offlinemorph.android.feature.aging

import android.graphics.Bitmap
import android.net.Uri

/** State of the aging inference pipeline. */
sealed interface AgingUiState {
    data object Idle : AgingUiState
    data class Loading(val message: String) : AgingUiState
    data class Success(val resultBitmap: Bitmap) : AgingUiState
    data class Error(val message: String) : AgingUiState
    /** Feature is disabled by the build-time flag. */
    data object Unavailable : AgingUiState
}

/** Full-screen state for the aging screen. */
data class AgingScreenState(
    val sourceUri: Uri? = null,
    val ageOffsetYears: Int = 0,
    val intensity: Float = 1.0f,
    val agingState: AgingUiState = AgingUiState.Idle,
    val isWorking: Boolean = false,
)
