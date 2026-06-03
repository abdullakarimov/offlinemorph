package com.offlinemorph.android.core.ml.ancestry

import android.graphics.Bitmap
import com.offlinemorph.android.core.ml.EngineProgress
import com.offlinemorph.android.core.ml.EngineResult

/**
 * Input request for the ancestry blend engine.
 *
 * Generates a hybrid portrait that projects features of [faceABitmap] and [faceBBitmap]
 * onto [targetBitmap] using a latent-blend of their ArcFace-style identity embeddings.
 *
 * @param targetBitmap  base portrait to apply the blended identity onto.
 * @param faceABitmap   first donor face.
 * @param faceBBitmap   second donor face.
 * @param blendRatio    weight of [faceABitmap] in [0f, 1f]; 1.0 = full face A, 0.0 = full face B.
 * @param intensity     overall effect strength in [0f, 1f].
 */
data class AncestryRequest(
    val targetBitmap: Bitmap,
    val faceABitmap: Bitmap,
    val faceBBitmap: Bitmap,
    val blendRatio: Float = 0.5f,
    val intensity: Float = 1.0f,
)

/**
 * Output produced by a successful ancestry blend run.
 *
 * @param outputBitmap   the blended portrait.
 * @param appliedRatio   the actual blend ratio used after clamping.
 */
data class AncestryResult(
    val outputBitmap: Bitmap,
    val appliedRatio: Float,
)

/** Contract for on-device ancestry blend engines. */
interface AncestryEngine {
    suspend fun runAncestry(
        request: AncestryRequest,
        onProgress: (EngineProgress) -> Unit = {},
    ): EngineResult<AncestryResult>
}
