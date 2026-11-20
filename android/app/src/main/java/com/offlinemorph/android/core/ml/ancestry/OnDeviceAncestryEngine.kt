package com.offlinemorph.android.core.ml.ancestry

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import com.offlinemorph.android.core.ml.EngineError
import com.offlinemorph.android.core.ml.EngineProgress
import com.offlinemorph.android.core.ml.EngineResult
import com.offlinemorph.android.core.ml.FaceAnalyzer
import com.offlinemorph.android.core.ml.FaceSwapEngine
import com.offlinemorph.android.core.ml.SwapRequest
import com.offlinemorph.android.feature.models.ModelCatalog
import java.io.File
import kotlin.math.sqrt

/**
 * Ancestry blend built from the core face-swap models — no extra download.
 *
 * The identity of each donor face is an ArcFace embedding (w600k_r50). The two embeddings are
 * mixed in identity space — `normalize(ratio·Â + (1 − ratio)·B̂)` — and the blended identity is
 * rendered onto [AncestryRequest.targetBitmap] by inswapper, which is an identity-conditioned
 * generator. [AncestryRequest.intensity] fades the result toward the untouched base portrait.
 *
 * This is the "latent blend" approach from IMPLEMENTATION_PLAN.md Phase C, with inswapper as
 * the generator.
 */
class OnDeviceAncestryEngine(
    private val modelsDirectory: File,
    private val faceAnalyzer: FaceAnalyzer,
    private val swapEngine: FaceSwapEngine,
    /** Runs the optional GFPGAN pass on the blended face when the model is installed. */
    private val enhanceWhenAvailable: Boolean = true,
) : AncestryEngine {

    override suspend fun runAncestry(
        request: AncestryRequest,
        onProgress: (EngineProgress) -> Unit,
    ): EngineResult<AncestryResult> {
        val missing = listOf(ModelCatalog.DETECTOR, ModelCatalog.RECOGNIZER, ModelCatalog.INSWAPPER)
            .firstOrNull { !File(modelsDirectory, it).isFile }
        if (missing != null) {
            return EngineResult.Failure(
                error = EngineError.ModelNotFound(missing),
                statusMessage = "Core face models are not installed. Open Setup → Download Required Models.",
            )
        }
        val ratio = request.blendRatio.coerceIn(0f, 1f)
        val intensity = request.intensity.coerceIn(0f, 1f)

        return try {
            onProgress(EngineProgress("Reading face A…", 0.05f))
            val embeddingA = faceAnalyzer.analyze(request.faceABitmap, includeEmbedding = true).embedding
                ?: return noFace("face A")
            onProgress(EngineProgress("Reading face B…", 0.2f))
            val embeddingB = faceAnalyzer.analyze(request.faceBBitmap, includeEmbedding = true).embedding
                ?: return noFace("face B")

            val blended = blendEmbeddings(embeddingA, embeddingB, ratio)
            onProgress(EngineProgress("Generating blended face…", 0.4f))
            val swap = swapEngine.runSwap(
                SwapRequest(
                    sourceBitmap = request.targetBitmap,
                    targetBitmap = request.targetBitmap,
                    sourceEmbedding = blended,
                    enhancerEnabled = enhanceWhenAvailable && File(modelsDirectory, ModelCatalog.GFPGAN).isFile,
                ),
            ) { message -> onProgress(EngineProgress(message, 0.6f)) }
            val swapped = swap.outputBitmap ?: return EngineResult.Failure(
                error = if (swap.statusMessage.startsWith("No face")) {
                    EngineError.NoFaceDetected("base portrait")
                } else {
                    EngineError.InferenceFailure(IllegalStateException(swap.statusMessage))
                },
                statusMessage = swap.statusMessage,
            )

            val output = if (intensity >= 1f) swapped else fade(request.targetBitmap, swapped, intensity)
            onProgress(EngineProgress("Done", 1f))
            EngineResult.Success(AncestryResult(outputBitmap = output, appliedRatio = ratio))
        } catch (e: Exception) {
            EngineResult.Failure(
                error = EngineError.InferenceFailure(e),
                statusMessage = e.message ?: "Ancestry blend failed.",
            )
        }
    }

    private fun noFace(stage: String) = EngineResult.Failure(
        error = EngineError.NoFaceDetected(stage),
        statusMessage = "No face found in $stage.",
    )

    /** Draws [effect] over [base] at [alpha]; both have the same size. */
    private fun fade(base: Bitmap, effect: Bitmap, alpha: Float): Bitmap {
        val out = base.copy(Bitmap.Config.ARGB_8888, true)
        Canvas(out).drawBitmap(effect, 0f, 0f, Paint().apply { this.alpha = (alpha * 255).toInt() })
        if (effect !== base) effect.recycle()
        return out
    }

    companion object {
        /**
         * Linear interpolation of the L2-normalised embeddings, re-normalised: the blend direction
         * on the identity hypersphere. [ratio] = 1 gives pure A, 0 gives pure B.
         */
        fun blendEmbeddings(a: FloatArray, b: FloatArray, ratio: Float): FloatArray {
            require(a.size == b.size) { "Embedding sizes differ: ${a.size} vs ${b.size}" }
            val na = normalize(a)
            val nb = normalize(b)
            return normalize(FloatArray(a.size) { ratio * na[it] + (1f - ratio) * nb[it] })
        }

        private fun normalize(v: FloatArray): FloatArray {
            val norm = sqrt(v.fold(0f) { acc, x -> acc + x * x })
            return if (norm <= 1e-12f) v.copyOf() else FloatArray(v.size) { v[it] / norm }
        }
    }
}
