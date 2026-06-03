package com.offlinemorph.android.core.ml.beautify

import com.offlinemorph.android.core.ml.EngineError
import com.offlinemorph.android.core.ml.EngineProgress
import com.offlinemorph.android.core.ml.EngineResult

/**
 * Stub [BeautifyEngine] used while the on-device beautification model is not yet bundled.
 *
 * Always returns [EngineResult.Failure] with [EngineError.ModelNotFound].
 */
class StubBeautifyEngine : BeautifyEngine {
    override suspend fun runBeautify(
        request: BeautifyRequest,
        onProgress: (EngineProgress) -> Unit,
    ): EngineResult<BeautifyResult> = EngineResult.Failure(
        error = EngineError.ModelNotFound("beautify_model.onnx"),
        statusMessage = "On-device beautification model is not yet available. Check back in a future update.",
    )
}
