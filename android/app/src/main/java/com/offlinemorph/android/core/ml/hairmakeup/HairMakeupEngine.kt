package com.offlinemorph.android.core.ml.hairmakeup

import android.graphics.Bitmap
import com.offlinemorph.android.core.ml.EngineProgress
import com.offlinemorph.android.core.ml.EngineResult

/**
 * Describes a desired hair and makeup transformation.
 *
 * @param sourceBitmap     portrait to apply the style onto.
 * @param hairColorHex     target hair colour in 6-digit RGB hex (e.g. "FF5733"), or null to skip.
 * @param lipColorHex      target lip colour in 6-digit RGB hex, or null to skip.
 * @param eyeshadowColorHex  target eye-shadow colour, or null to skip.
 * @param intensity        overall effect strength in [0f, 1f].
 */
data class HairMakeupRequest(
    val sourceBitmap: Bitmap,
    val hairColorHex: String? = null,
    val lipColorHex: String? = null,
    val eyeshadowColorHex: String? = null,
    val intensity: Float = 1.0f,
)

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
