package com.offlinemorph.android.core.ml.beautify

import android.graphics.Bitmap
import com.offlinemorph.android.core.ml.EngineError
import com.offlinemorph.android.core.ml.EngineProgress
import com.offlinemorph.android.core.ml.EngineResult
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

/**
 * On-device beautification engine using OpenCV image processing only — no ONNX model required.
 *
 * Currently implements:
 *  - [BeautifyRequest.skinSmoothing] via edge-preserving bilateral filter.
 *
 * [BeautifyRequest.eyeEnlarge], [BeautifyRequest.faceSlim], and [BeautifyRequest.teethWhiten]
 * require landmark-based warping or segmentation models not yet available; those controls are
 * accepted in the request but do not currently alter the output.
 */
class OnDeviceBeautifyEngine : BeautifyEngine {

    override suspend fun runBeautify(
        request: BeautifyRequest,
        onProgress: (EngineProgress) -> Unit,
    ): EngineResult<BeautifyResult> {
        return try {
            onProgress(EngineProgress("Applying skin smoothing…", 0.1f))
            val output = applyEnhancements(request)
            onProgress(EngineProgress("Done", 1f))
            EngineResult.Success(BeautifyResult(output))
        } catch (e: Exception) {
            EngineResult.Failure(
                error = EngineError.InferenceFailure(e),
                statusMessage = e.message ?: "Beautify processing failed.",
            )
        }
    }

    private fun applyEnhancements(request: BeautifyRequest): Bitmap {
        if (request.skinSmoothing <= 0f) {
            return request.sourceBitmap.copy(Bitmap.Config.ARGB_8888, false)
        }

        // Work at a capped resolution so bilateral filter stays responsive on-device.
        val src = request.sourceBitmap
        val maxPx = 1024
        val scale = minOf(1f, maxPx.toFloat() / maxOf(src.width, src.height))
        val workW = (src.width * scale).toInt().coerceAtLeast(1)
        val workH = (src.height * scale).toInt().coerceAtLeast(1)
        val work = if (scale < 1f) Bitmap.createScaledBitmap(src, workW, workH, true) else src

        // RGBA → RGB for bilateral filter (4-channel not supported by bilateralFilter).
        val rgbaSrc = Mat()
        Utils.bitmapToMat(work, rgbaSrc)
        if (work !== src) work.recycle()

        val rgbSrc = Mat()
        Imgproc.cvtColor(rgbaSrc, rgbSrc, Imgproc.COLOR_RGBA2RGB)
        rgbaSrc.release()

        // Bilateral filter: sigma scales with intensity. d=-1 → size derived from sigmaSpace.
        val sigma = (request.skinSmoothing * 60.0).coerceAtLeast(5.0)
        val filtered = Mat()
        Imgproc.bilateralFilter(rgbSrc, filtered, -1, sigma, sigma)

        // Blend filtered with original by intensity so low values apply a subtle effect.
        val blended = Mat()
        Core.addWeighted(filtered, request.skinSmoothing.toDouble(), rgbSrc, (1.0 - request.skinSmoothing), 0.0, blended)
        rgbSrc.release()
        filtered.release()

        val rgbaDst = Mat()
        Imgproc.cvtColor(blended, rgbaDst, Imgproc.COLOR_RGB2RGBA)
        blended.release()

        val resultBitmap = Bitmap.createBitmap(workW, workH, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(rgbaDst, resultBitmap)
        rgbaDst.release()

        return if (scale < 1f) {
            Bitmap.createScaledBitmap(resultBitmap, src.width, src.height, true)
                .also { resultBitmap.recycle() }
        } else {
            resultBitmap
        }
    }
}

