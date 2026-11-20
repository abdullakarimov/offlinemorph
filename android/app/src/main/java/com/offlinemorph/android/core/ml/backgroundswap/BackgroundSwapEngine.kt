package com.offlinemorph.android.core.ml.backgroundswap

import android.graphics.Bitmap
import com.offlinemorph.android.core.ml.EngineProgress
import com.offlinemorph.android.core.ml.EngineResult

/** How the matted subject is composited. */
sealed interface BackgroundMode {
    /** Replace the background with [image], scaled to cover the frame. */
    data class Image(val image: Bitmap) : BackgroundMode
    /** Keep the original background but blur it; [strength] in [0f, 1f]. */
    data class Bokeh(val strength: Float = 0.6f) : BackgroundMode
    /** Fill the background with a solid colour (packed ARGB). */
    data class SolidColor(val colorArgb: Int) : BackgroundMode
    /** Return only the subject over a transparent background. */
    data object Transparent : BackgroundMode
}

/**
 * Input request for the background swap engine (Phase G).
 *
 * @param sourceBitmap   photo containing the subject.
 * @param mode           compositing mode for the new background.
 * @param edgeRefinement anti-halo / hair-detail refinement strength in [0f, 1f].
 * @param matchLighting  when true, the subject's colour is harmonised to the new background.
 */
data class BackgroundSwapRequest(
    val sourceBitmap: Bitmap,
    val mode: BackgroundMode,
    val edgeRefinement: Float = 0.5f,
    val matchLighting: Boolean = true,
)

/**
 * Output produced by a successful background swap.
 *
 * @param outputBitmap the composited image.
 * @param alphaMatte   the subject matte (ALPHA_8 or grayscale ARGB, 255 = subject), reusable for
 *                     re-compositing with a different background without re-running the model.
 */
data class BackgroundSwapResult(
    val outputBitmap: Bitmap,
    val alphaMatte: Bitmap,
)

/** Contract for on-device background matting and replacement engines. */
interface BackgroundSwapEngine {
    suspend fun runBackgroundSwap(
        request: BackgroundSwapRequest,
        onProgress: (EngineProgress) -> Unit = {},
    ): EngineResult<BackgroundSwapResult>
}
