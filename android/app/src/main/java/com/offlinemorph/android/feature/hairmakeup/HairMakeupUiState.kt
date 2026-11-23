package com.offlinemorph.android.feature.hairmakeup

import android.graphics.Bitmap
import android.net.Uri
import com.offlinemorph.android.core.ml.hairmakeup.FacialHairStyle
import com.offlinemorph.android.core.ml.hairmakeup.HairStyle

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
    val hairStyle: HairStyle = HairStyle.SOLID,
    /** Tips (ombré) / streak (highlights) colour; null = lighter shade of the hair colour. */
    val hairSecondaryColorHex: String? = null,
    val lipColorHex: String? = null,
    val eyeshadowColorHex: String? = null,
    val eyelinerColorHex: String? = null,
    val blushColorHex: String? = null,
    val facialHair: FacialHairStyle = FacialHairStyle.NONE,
    /** Null = matched to the eyebrows. */
    val facialHairColorHex: String? = null,
    val intensity: Float = 0.8f,
    val hairMakeupState: HairMakeupUiState = HairMakeupUiState.Idle,
    val isWorking: Boolean = false,
) {
    val hasAnyEffect: Boolean
        get() = hairColorHex != null || lipColorHex != null || eyeshadowColorHex != null ||
            eyelinerColorHex != null || blushColorHex != null || facialHair != FacialHairStyle.NONE
}
