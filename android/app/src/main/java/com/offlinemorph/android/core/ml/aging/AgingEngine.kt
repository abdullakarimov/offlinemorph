package com.offlinemorph.android.core.ml.aging

import android.graphics.Bitmap
import com.offlinemorph.android.core.ml.EngineProgress
import com.offlinemorph.android.core.ml.EngineResult

/**
 * Input request for the aging / de-aging engine.
 *
 * @param sourceBitmap    portrait bitmap to transform.
 * @param ageOffsetYears  signed age delta in years. Negative = de-aging, positive = aging.
 *                        Practical range is [-50, +50]; values outside this range are clamped.
 * @param intensity       realism blend factor in [0f, 1f]. 1.0 applies the full transformation;
 *                        lower values blend toward the original image.
 */
data class AgingRequest(
    val sourceBitmap: Bitmap,
    val ageOffsetYears: Int,
    val intensity: Float = 1.0f,
)

/**
 * Output produced by a successful aging run.
 *
 * @param outputBitmap    the transformed portrait.
 * @param appliedAgeOffset the actual age offset used after clamping.
 */
data class AgingResult(
    val outputBitmap: Bitmap,
    val appliedAgeOffset: Int,
)

/** Contract for on-device aging / de-aging engines. */
interface AgingEngine {
    suspend fun runAging(
        request: AgingRequest,
        onProgress: (EngineProgress) -> Unit = {},
    ): EngineResult<AgingResult>
}
