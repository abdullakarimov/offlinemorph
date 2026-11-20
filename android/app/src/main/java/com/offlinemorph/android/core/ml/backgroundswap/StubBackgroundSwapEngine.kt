package com.offlinemorph.android.core.ml.backgroundswap

import com.offlinemorph.android.core.ml.EngineError
import com.offlinemorph.android.core.ml.EngineProgress
import com.offlinemorph.android.core.ml.EngineResult

/**
 * Stub [BackgroundSwapEngine] used until the portrait matting model is integrated.
 *
 * Always returns [EngineResult.Failure] with [EngineError.ModelNotFound].
 */
class StubBackgroundSwapEngine : BackgroundSwapEngine {
    override suspend fun runBackgroundSwap(
        request: BackgroundSwapRequest,
        onProgress: (EngineProgress) -> Unit,
    ): EngineResult<BackgroundSwapResult> = EngineResult.Failure(
        error = EngineError.ModelNotFound("portrait_matting.onnx"),
        statusMessage = "On-device background swap is not yet available. Check back in a future update.",
    )
}
