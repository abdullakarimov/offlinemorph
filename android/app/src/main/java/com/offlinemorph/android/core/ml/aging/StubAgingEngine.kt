package com.offlinemorph.android.core.ml.aging

import com.offlinemorph.android.core.ml.EngineError
import com.offlinemorph.android.core.ml.EngineProgress
import com.offlinemorph.android.core.ml.EngineResult

/**
 * Stub [AgingEngine] returned while the on-device aging model is not yet available.
 *
 * Always returns [EngineResult.Failure] with [EngineError.ModelNotFound] so that feature
 * flag-gated callers can surface a graceful "coming soon" state without a crash path.
 */
class StubAgingEngine : AgingEngine {
    override suspend fun runAging(
        request: AgingRequest,
        onProgress: (EngineProgress) -> Unit,
    ): EngineResult<AgingResult> = EngineResult.Failure(
        error = EngineError.ModelNotFound("aging_model.onnx"),
        statusMessage = "On-device aging model is not yet available. Check back in a future update.",
    )
}
