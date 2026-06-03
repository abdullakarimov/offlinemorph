package com.offlinemorph.android.core.ml.hairmakeup

import com.offlinemorph.android.core.ml.EngineError
import com.offlinemorph.android.core.ml.EngineProgress
import com.offlinemorph.android.core.ml.EngineResult

/**
 * Stub [HairMakeupEngine] used while the on-device hair/makeup model is not yet bundled.
 *
 * Always returns [EngineResult.Failure] with [EngineError.ModelNotFound].
 */
class StubHairMakeupEngine : HairMakeupEngine {
    override suspend fun runHairMakeup(
        request: HairMakeupRequest,
        onProgress: (EngineProgress) -> Unit,
    ): EngineResult<HairMakeupResult> = EngineResult.Failure(
        error = EngineError.ModelNotFound("hairmakeup_model.onnx"),
        statusMessage = "On-device hair & makeup model is not yet available. Check back in a future update.",
    )
}
