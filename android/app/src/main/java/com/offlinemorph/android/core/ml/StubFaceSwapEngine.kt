package com.offlinemorph.android.core.ml

class StubFaceSwapEngine : FaceSwapEngine {
    private val preprocessor = InSwapperImagePreprocessor()
    private val postprocessor = InSwapperImagePostprocessor()

    override suspend fun runSwap(request: SwapRequest, onProgress: (String) -> Unit): SwapRunResult {
        val preparedTarget = preprocessor.prepareTarget(request.targetBitmap)
        val previewBitmap = postprocessor.createPreviewBitmap(preparedTarget.resizedBitmap)

        return SwapRunResult(
            statusMessage = "Prepared target tensor for inswapper-style inference at ${preparedTarget.inputSize}x${preparedTarget.inputSize}. Next step is wiring source embedding projection and ONNX session execution.",
            outputBitmap = previewBitmap,
        )
    }
}

