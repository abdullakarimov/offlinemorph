package com.offlinemorph.android.core.ml.ancestry

import com.offlinemorph.android.core.ml.EngineError
import com.offlinemorph.android.core.ml.EngineProgress
import com.offlinemorph.android.core.ml.EngineResult

/**
 * Stub [AncestryEngine] used while the on-device ancestry model is not yet available.
 *
 * Always returns [EngineResult.Failure] with [EngineError.ModelNotFound].
 */
class StubAncestryEngine : AncestryEngine {
    override suspend fun runAncestry(
        request: AncestryRequest,
        onProgress: (EngineProgress) -> Unit,
    ): EngineResult<AncestryResult> = EngineResult.Failure(
        error = EngineError.ModelNotFound("ancestry_model.onnx"),
        statusMessage = "On-device ancestry blend model is not yet available. Check back in a future update.",
    )
}
