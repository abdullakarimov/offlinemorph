package com.offlinemorph.android.core.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.graphics.Bitmap
import android.util.Log
import com.offlinemorph.android.feature.models.ModelCatalog
import java.io.File
import kotlin.math.sqrt
import org.opencv.core.CvType
import org.opencv.core.Mat

class OnDeviceFaceSwapEngine(
    private val modelsDirectory: File,
    private val faceAnalyzer: FaceAnalyzer,
    private val sessionFactory: OrtSessionFactory,
) : FaceSwapEngine {
    private val enhancer = OnDeviceFaceEnhancer(modelsDirectory, sessionFactory)

    /** Cached emap extracted from inswapper_128.onnx on first swap run. */
    @Volatile private var cachedEmap: FloatArray? = null

    /** Latent for the last source embedding; a video swap reuses one source for every frame. */
    @Volatile private var cachedLatent: Pair<FloatArray, FloatArray>? = null

    private fun getEmap(modelFile: File): FloatArray? {
        cachedEmap?.let { return it }
        return synchronized(this) {
            cachedEmap ?: runCatching {
                EmapExtractor.getEmap(modelFile).also {
                    cachedEmap = it
                    Log.d(TAG, "emap extracted size=${it.size}")
                }
            }.getOrElse { e ->
                Log.e(TAG, "emap extraction failed", e)
                null
            }
        }
    }

    override suspend fun runSwap(request: SwapRequest, onProgress: (String) -> Unit): SwapRunResult {
        val startNs = System.nanoTime()
        onProgress("Analyzing Faces...")
        // Apply the execution-policy image size cap before any inference.
        val scaledRequest = if (request.maxImageSizePx > 0) request.scaledTo(request.maxImageSizePx) else request
        val needsGender = request.faceFilterMode == FaceFilterMode.MALE_ONLY ||
            request.faceFilterMode == FaceFilterMode.FEMALE_ONLY
        val sourceAnalysis = faceAnalyzer.analyze(scaledRequest.sourceBitmap, includeEmbedding = true)
        val targetAnalysis = faceAnalyzer.analyze(scaledRequest.targetBitmap, includeGender = needsGender)
        val inswapperFile = File(modelsDirectory, ModelCatalog.INSWAPPER)

        if (!inswapperFile.isFile) {
            return SwapRunResult(statusMessage = "Missing swap model: ${ModelCatalog.INSWAPPER}")
        }

        val sourceEmbedding = sourceAnalysis.embedding
        if (sourceEmbedding == null || sourceEmbedding.isEmpty()) {
            return SwapRunResult(statusMessage = "No face found in the source image. ${sourceAnalysis.statusMessage}")
        }

        val targetKeypoints = selectTargetKeypoints(targetAnalysis, request)
        if (targetKeypoints.isEmpty()) {
            return SwapRunResult(
                statusMessage = if (request.faceFilterMode == FaceFilterMode.SPECIFIC) {
                    "No face found in the target image. ${targetAnalysis.statusMessage}"
                } else {
                    "No faces matched the '${request.faceFilterMode.displayName}' filter."
                },
            )
        }

        return runCatching {
            val session = sessionFactory.session(inswapperFile)
            val latent = latentFor(sourceEmbedding, inswapperFile, session)
            val boost = request.pixelBoost.coerceIn(1, 4)

            // All faces share one analysis: swapping only changes texture, never face positions.
            var output = scaledRequest.targetBitmap
            for ((i, kps) in targetKeypoints.withIndex()) {
                onProgress(
                    if (targetKeypoints.size > 1) "Swapping face ${i + 1} / ${targetKeypoints.size}..."
                    else "Running Inswapper...",
                )
                output = replacing(output, scaledRequest.targetBitmap) {
                    swapFace(session, output, kps, latent, boost, request.seamlessBlend)
                }
                if (request.enhancerEnabled) {
                    onProgress("Enhancing Facial Details...")
                    output = replacing(output, scaledRequest.targetBitmap) {
                        enhancer.enhanceFace(output, kps) ?: output
                    }
                }
            }
            Log.d(TAG, "Swapped ${targetKeypoints.size} face(s) boost=$boost enhancer=${request.enhancerEnabled} " +
                "in ${(System.nanoTime() - startNs) / 1_000_000} ms")
            SwapRunResult(
                statusMessage = "Swapped ${targetKeypoints.size} face(s).",
                outputBitmap = output,
            )
        }.getOrElse { error ->
            Log.e(TAG, "Inswapper run failed", error)
            SwapRunResult(statusMessage = "Face swap failed: ${error.message}")
        }
    }

    /** Runs [step] and recycles the previous intermediate bitmap once it has been superseded. */
    private inline fun replacing(previous: Bitmap, original: Bitmap, step: () -> Bitmap): Bitmap {
        val next = step()
        if (next !== previous && previous !== original) previous.recycle()
        return next
    }

    private fun selectTargetKeypoints(analysis: FaceAnalysisSummary, request: SwapRequest): List<FloatArray> {
        val faces = analysis.allDetectedFaces
        val selected = when (request.faceFilterMode) {
            FaceFilterMode.SPECIFIC -> listOfNotNull(faces.getOrNull(request.targetFaceIndex) ?: faces.firstOrNull())
            FaceFilterMode.ALL_FACES -> faces
            FaceFilterMode.MALE_ONLY -> faces.filter { it.isMale == true }
            FaceFilterMode.FEMALE_ONLY -> faces.filter { it.isMale == false }
        }
        val keypoints = selected.mapNotNull { face ->
            face.fiveKeypoints ?: analysis.landmarkFivePoints?.takeIf { face.index == 0 }
        }
        // Non-SCRFD detector outputs only report the primary face.
        val primaryOnly = request.faceFilterMode == FaceFilterMode.SPECIFIC ||
            request.faceFilterMode == FaceFilterMode.ALL_FACES
        return if (keypoints.isEmpty() && faces.isEmpty() && primaryOnly) {
            listOfNotNull(analysis.landmarkFivePoints)
        } else {
            keypoints
        }
    }

    /**
     * Swaps one face: align the target with the inswapper template, run the model (N² times
     * with pixel boost), and paste the result back.
     */
    private fun swapFace(
        session: OrtSession,
        target: Bitmap,
        keypoints: FloatArray,
        latent: FloatArray,
        boost: Int,
        seamless: Boolean,
    ): Bitmap {
        val size = FaceAlignmentOps.ALIGNED_SIZE
        val cropSize = size * boost
        val toCrop = FaceAlignmentOps.alignmentMatrix(keypoints, FaceTemplate.ARCFACE_128, cropSize)
        val crop = FaceAlignmentOps.warpToCrop(target, toCrop, cropSize)
        val cropBytes = ByteArray(cropSize * cropSize * 3).also { crop.get(0, 0, it) }
        crop.release()

        val targetName = if ("target" in session.inputNames) "target" else session.inputNames.first()
        val sourceName = if ("source" in session.inputNames) "source" else session.inputNames.drop(1).first()
        val plane = size * size
        val input = FloatArray(plane * 3)
        val outBytes = ByteArray(cropBytes.size)

        sessionFactory.createFloatTensor(latent, longArrayOf(1, latent.size.toLong())).use { sourceTensor ->
            for (dy in 0 until boost) for (dx in 0 until boost) {
                // Each pass sees every boost-th pixel: a regular 128 px aligned crop shifted by a sub-pixel offset.
                for (y in 0 until size) for (x in 0 until size) {
                    val src = ((y * boost + dy) * cropSize + (x * boost + dx)) * 3
                    val i = y * size + x
                    input[i] = (cropBytes[src].toInt() and 0xFF) / 255f
                    input[plane + i] = (cropBytes[src + 1].toInt() and 0xFF) / 255f
                    input[plane * 2 + i] = (cropBytes[src + 2].toInt() and 0xFF) / 255f
                }
                val swapped = runInswapper(session, targetName, sourceName, input, sourceTensor)
                for (y in 0 until size) for (x in 0 until size) {
                    val dst = ((y * boost + dy) * cropSize + (x * boost + dx)) * 3
                    val i = y * size + x
                    outBytes[dst] = toByte(swapped[i])
                    outBytes[dst + 1] = toByte(swapped[plane + i])
                    outBytes[dst + 2] = toByte(swapped[plane * 2 + i])
                }
            }
        }

        val patch = Mat(cropSize, cropSize, CvType.CV_8UC3).apply { put(0, 0, outBytes) }
        try {
            return FaceAlignmentOps.pasteBack(target, patch, FaceAlignmentOps.invert(toCrop), seamless)
        } finally {
            patch.release()
        }
    }

    private fun runInswapper(
        session: OrtSession,
        targetName: String,
        sourceName: String,
        input: FloatArray,
        sourceTensor: OnnxTensor,
    ): FloatArray {
        val size = FaceAlignmentOps.ALIGNED_SIZE.toLong()
        return sessionFactory.createFloatTensor(input, longArrayOf(1, 3, size, size)).use { targetTensor ->
            session.run(mapOf(targetName to targetTensor, sourceName to sourceTensor)).use { result ->
                val outputs = OrtValueUtils.extractFloatOutputs(result)
                val selected = selectSwapOutput(outputs, FaceAlignmentOps.ALIGNED_SIZE)
                    ?: error("Inswapper returned no outputs")
                toChwTensor(selected, FaceAlignmentOps.ALIGNED_SIZE)
                    ?: error("Unexpected inswapper output ${selected.name} of size ${selected.data.size}")
            }
        }
    }

    private fun toByte(v: Float): Byte = (v * 255f + 0.5f).toInt().coerceIn(0, 255).toByte()

    private fun latentFor(embedding: FloatArray, modelFile: File, session: OrtSession): FloatArray {
        cachedLatent?.let { (cachedEmbedding, latent) -> if (cachedEmbedding === embedding) return latent }
        val sourceName = if ("source" in session.inputNames) "source" else session.inputNames.drop(1).firstOrNull()
        val expectedLength = (session.inputInfo[sourceName]?.info as? TensorInfo)
            ?.shape?.lastOrNull()?.toInt()?.takeIf { it > 0 }
        // Python: latent = normed_embedding @ emap; latent /= norm(latent)
        val emap = getEmap(modelFile)
        val latent = if (emap != null) {
            projectEmbeddingThroughEmap(embedding, emap, expectedLength ?: 512)
        } else {
            prepareCompatibleEmbedding(embedding, expectedLength)
        }
        Log.d(TAG, "latent norm=${sqrt(latent.fold(0f) { acc, v -> acc + v * v })} emap=${emap != null}")
        cachedLatent = embedding to latent
        return latent
    }

    /**
     * Projects the raw ArcFace embedding through the inswapper's embedding map matrix.
     *
     * Python equivalent:
     *   normed = embedding / norm(embedding)       # L2 normalise
     *   latent = normed.reshape(1,-1) @ emap       # matrix multiply, shape (1, emapCols)
     *   latent /= norm(latent)                     # L2 normalise result
     *
     * The emap is stored in row-major order: element [i,j] is at emap[i*emapCols + j].
     */
    private fun projectEmbeddingThroughEmap(
        rawEmbedding: FloatArray,
        emap: FloatArray,
        emapCols: Int,
    ): FloatArray {
        val emapRows = emap.size / emapCols
        val normed = l2Normalize(rawEmbedding)
        val inputLen = minOf(normed.size, emapRows)
        val result = FloatArray(emapCols)
        for (j in 0 until emapCols) {
            var sum = 0f
            for (i in 0 until inputLen) {
                sum += normed[i] * emap[i * emapCols + j]
            }
            result[j] = sum
        }
        return l2Normalize(result)
    }

    private fun prepareCompatibleEmbedding(rawEmbedding: FloatArray, expectedLength: Int?): FloatArray {
        val normalized = l2Normalize(rawEmbedding)
        if (expectedLength == null || expectedLength <= 0) {
            return normalized
        }

        if (normalized.size == expectedLength) {
            return normalized
        }

        val out = FloatArray(expectedLength)
        if (normalized.size >= expectedLength) {
            normalized.copyInto(out, 0, 0, expectedLength)
        } else {
            normalized.copyInto(out, 0, 0, normalized.size)
        }
        return l2Normalize(out)
    }

    private fun l2Normalize(vector: FloatArray): FloatArray {
        var sumSq = 0.0f
        for (v in vector) {
            sumSq += v * v
        }
        if (sumSq <= 1e-12f) {
            return vector.copyOf()
        }
        val inv = 1.0f / sqrt(sumSq)
        val out = FloatArray(vector.size)
        for (i in vector.indices) {
            out[i] = vector[i] * inv
        }
        return out
    }

    private fun selectSwapOutput(outputs: List<TensorData>, expectedSize: Int): TensorData? {
        if (outputs.isEmpty()) {
            return null
        }

        val expectedFlat = expectedSize * expectedSize * 3
        return outputs.maxWithOrNull(
            compareByDescending<TensorData> { swapOutputScore(it, expectedSize, expectedFlat) }
                .thenByDescending { it.data.size }
        )
    }

    private fun swapOutputScore(output: TensorData, expectedSize: Int, expectedFlat: Int): Int {
        var score = 0
        val name = output.name
        if (name.contains("output", ignoreCase = true) || name.contains("image", ignoreCase = true)) {
            score += 20
        }

        val shape = output.shape
        if (shape != null && shape.size == 4) {
            val n = shape[0].toInt()
            val d1 = shape[1].toInt()
            val d2 = shape[2].toInt()
            val d3 = shape[3].toInt()

            if (n == 1) {
                score += 10
            }

            val nchwMatch = d1 == 3 && d2 == expectedSize && d3 == expectedSize
            val nhwcMatch = d1 == expectedSize && d2 == expectedSize && d3 == 3
            if (nchwMatch || nhwcMatch) {
                score += 120
            } else if ((d1 == 3 || d3 == 3) && (d2 == expectedSize || d1 == expectedSize)) {
                score += 40
            }
        }

        val sizeDelta = kotlin.math.abs(output.data.size - expectedFlat)
        score += when {
            output.data.size == expectedFlat -> 100
            sizeDelta <= expectedSize * 8 -> 30
            else -> 0
        }

        return score
    }

    private fun toChwTensor(output: TensorData, expectedSize: Int): FloatArray? {
        val expectedFlat = expectedSize * expectedSize * 3
        if (output.data.size != expectedFlat) {
            return null
        }

        val shape = output.shape
        if (shape != null && shape.size == 4) {
            val nhwc = shape[1].toInt() == expectedSize && shape[2].toInt() == expectedSize && shape[3].toInt() == 3
            if (nhwc) {
                val chw = FloatArray(expectedFlat)
                val plane = expectedSize * expectedSize
                var i = 0
                var p = 0
                while (i < output.data.size) {
                    val r = output.data[i]
                    val g = output.data[i + 1]
                    val b = output.data[i + 2]
                    chw[p] = r
                    chw[plane + p] = g
                    chw[plane * 2 + p] = b
                    p += 1
                    i += 3
                }
                return chw
            }
        }
        return output.data
    }

    private companion object {
        const val TAG = "OFFLINEMORPH_SWAP"
    }
}
