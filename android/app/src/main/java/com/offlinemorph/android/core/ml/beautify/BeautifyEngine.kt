package com.offlinemorph.android.core.ml.beautify

import android.graphics.Bitmap
import com.offlinemorph.android.core.ml.EngineProgress
import com.offlinemorph.android.core.ml.EngineResult

/**
 * Describes a portrait beautification pass.
 *
 * Controls are independent; set a value to 0f to skip that enhancement.
 *
 * @param sourceBitmap  portrait to enhance.
 * @param skinSmoothing skin-retouching intensity in [0f, 1f].
 * @param eyeEnlarge    eye enlargement in [0f, 1f].
 * @param faceSlim      face-slimming in [0f, 1f].
 * @param teethWhiten   teeth-whitening in [0f, 1f].
 */
data class BeautifyRequest(
    val sourceBitmap: Bitmap,
    val skinSmoothing: Float = 0f,
    val eyeEnlarge: Float = 0f,
    val faceSlim: Float = 0f,
    val teethWhiten: Float = 0f,
)

/**
 * Result of a successful beautification pass.
 *
 * @param outputBitmap the enhanced portrait.
 */
data class BeautifyResult(
    val outputBitmap: Bitmap,
)

/** Contract for on-device portrait beautification engines. */
interface BeautifyEngine {
    suspend fun runBeautify(
        request: BeautifyRequest,
        onProgress: (EngineProgress) -> Unit = {},
    ): EngineResult<BeautifyResult>
}
