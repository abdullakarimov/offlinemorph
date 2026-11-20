package com.offlinemorph.android.core.ml.relight

import com.offlinemorph.android.core.ml.EngineError
import com.offlinemorph.android.core.ml.EngineProgress
import com.offlinemorph.android.core.ml.EngineResult

/**
 * Stub [RelightEngine] used until the depth/normal estimation model is integrated.
 *
 * Always returns [EngineResult.Failure] with [EngineError.ModelNotFound].
 */
class StubRelightEngine : RelightEngine {
    override suspend fun runRelight(
        request: RelightRequest,
        onProgress: (EngineProgress) -> Unit,
    ): EngineResult<RelightResult> = EngineResult.Failure(
        error = EngineError.ModelNotFound("relight_depth.onnx"),
        statusMessage = "On-device relighting is not yet available. Check back in a future update.",
    )
}
