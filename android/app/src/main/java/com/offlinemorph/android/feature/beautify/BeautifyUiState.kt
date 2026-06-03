package com.offlinemorph.android.feature.beautify

import android.graphics.Bitmap
import android.net.Uri

sealed interface BeautifyUiState {
    data object Idle : BeautifyUiState
    data class Loading(val message: String) : BeautifyUiState
    data class Success(val resultBitmap: Bitmap) : BeautifyUiState
    data class Error(val message: String) : BeautifyUiState
    data object Unavailable : BeautifyUiState
}

data class BeautifyScreenState(
    val sourceUri: Uri? = null,
    val skinSmoothing: Float = 0.5f,
    val eyeEnlarge: Float = 0f,
    val faceSlim: Float = 0f,
    val teethWhiten: Float = 0f,
    val beautifyState: BeautifyUiState = BeautifyUiState.Idle,
    val isWorking: Boolean = false,
)
