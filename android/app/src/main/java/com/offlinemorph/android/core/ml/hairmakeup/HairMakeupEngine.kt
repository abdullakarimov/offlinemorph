package com.offlinemorph.android.core.ml.hairmakeup

import android.graphics.Bitmap
import com.offlinemorph.android.core.ml.EngineProgress
import com.offlinemorph.android.core.ml.EngineResult

/** How the hair colour is laid out. */
enum class HairStyle(val displayName: String) {
    /** One even colour. */
    SOLID("Solid"),
    /** Roots in the hair colour fading to the secondary colour at the tips. */
    OMBRE("Ombré"),
    /** Hair colour with streaks of the secondary colour. */
    HIGHLIGHTS("Highlights"),
}

/** Simulated facial hair. */
enum class FacialHairStyle(val displayName: String) {
    NONE("None"),
    STUBBLE("Stubble"),
    MUSTACHE("Mustache"),
    GOATEE("Goatee"),
    FULL_BEARD("Full Beard"),
}

/**
 * Describes a desired hair and makeup transformation. Colours are 6-digit RGB hex strings
 * (e.g. "FF5733"); null skips that effect.
 *
 * @param sourceBitmap          portrait to apply the style onto.
 * @param hairColorHex          hair colour (roots for [HairStyle.OMBRE], base for [HairStyle.HIGHLIGHTS]).
 * @param hairStyle             colour layout.
 * @param hairSecondaryColorHex tips / streak colour; null = a lighter shade of [hairColorHex].
 * @param lipColorHex           lipstick.
 * @param eyeshadowColorHex     eyelid shadow.
 * @param eyelinerColorHex      upper lash-line liner.
 * @param blushColorHex         cheek blush.
 * @param facialHair            facial hair to simulate.
 * @param facialHairColorHex    facial hair colour; null = matched to the eyebrows.
 * @param intensity             overall effect strength in [0f, 1f].
 */
data class HairMakeupRequest(
    val sourceBitmap: Bitmap,
    val hairColorHex: String? = null,
    val hairStyle: HairStyle = HairStyle.SOLID,
    val hairSecondaryColorHex: String? = null,
    val lipColorHex: String? = null,
    val eyeshadowColorHex: String? = null,
    val eyelinerColorHex: String? = null,
    val blushColorHex: String? = null,
    val facialHair: FacialHairStyle = FacialHairStyle.NONE,
    val facialHairColorHex: String? = null,
    val intensity: Float = 1.0f,
) {
    val hasAnyEffect: Boolean
        get() = hairColorHex != null || lipColorHex != null || eyeshadowColorHex != null ||
            eyelinerColorHex != null || blushColorHex != null || facialHair != FacialHairStyle.NONE
}

/**
 * Result of a successful hair and makeup transformation.
 *
 * @param outputBitmap the styled portrait.
 */
data class HairMakeupResult(
    val outputBitmap: Bitmap,
)

/** Contract for on-device hair and makeup synthesis engines. */
interface HairMakeupEngine {
    suspend fun runHairMakeup(
        request: HairMakeupRequest,
        onProgress: (EngineProgress) -> Unit = {},
    ): EngineResult<HairMakeupResult>
}
