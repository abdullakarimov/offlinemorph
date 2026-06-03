package com.offlinemorph.android.feature.hairmakeup

import android.graphics.Bitmap
import android.net.Uri

sealed interface HairMakeupUiState {
    data object Idle : HairMakeupUiState
    data class Loading(val message: String) : HairMakeupUiState
    data class Success(val resultBitmap: Bitmap) : HairMakeupUiState
    data class Error(val message: String) : HairMakeupUiState
    data object Unavailable : HairMakeupUiState
    /** Feature is enabled but face_parsing.onnx is not yet downloaded. */
    data object ModelNotReady : HairMakeupUiState
}

data class HairMakeupScreenState(
    val sourceUri: Uri? = null,
    val hairColorHex: String? = null,
    val lipColorHex: String? = null,
    val intensity: Float = 0.8f,
    val hairMakeupState: HairMakeupUiState = HairMakeupUiState.Idle,
    val isWorking: Boolean = false,
)
