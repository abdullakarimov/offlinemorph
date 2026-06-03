package com.offlinemorph.android.core.ml

import ai.onnxruntime.TensorInfo
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.util.Log
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import com.offlinemorph.android.feature.models.ModelCatalog
import java.io.File
import java.nio.FloatBuffer

class OnDeviceFaceSwapEngine(
    private val modelsDirectory: File,
    private val faceAnalyzer: FaceAnalyzer,
    private val sessionFactory: OrtSessionFactory,
) : FaceSwapEngine {
    private val preprocessor = InSwapperImagePreprocessor()
    private val postprocessor = InSwapperImagePostprocessor()
    private val enhancer = OnDeviceFaceEnhancer(modelsDirectory)

    /** Cached emap extracted from inswapper_128.onnx on first swap run. */
    @Volatile private var cachedEmap: FloatArray? = null

    private fun getEmap(modelFile: File): FloatArray? {
        cachedEmap?.let { return it }
        return synchronized(this) {
            cachedEmap ?: runCatching {
                EmapExtractor.getEmap(modelFile).also {
                    cachedEmap = it
                    Log.d("OFFLINEMORPH_SWAP", "emap extracted size=${it.size}")
                }
            }.getOrElse { e ->
                Log.e("OFFLINEMORPH_SWAP", "emap extraction failed", e)
                null
            }
        }
    }

    override suspend fun runSwap(request: SwapRequest, onProgress: (String) -> Unit): SwapRunResult {
        onProgress("Analyzing Faces...")
        val sourceAnalysis = faceAnalyzer.analyze(request.sourceBitmap)
        val targetAnalysis = faceAnalyzer.analyze(request.targetBitmap)
        val inswapperFile = File(modelsDirectory, ModelCatalog.INSWAPPER)

        if (!inswapperFile.isFile) {
            return SwapRunResult(
                statusMessage = "${targetAnalysis.statusMessage} Missing swap model: ${ModelCatalog.INSWAPPER}",
            )
        }

        val sourceEmbedding = sourceAnalysis.embedding
        if (sourceEmbedding == null || sourceEmbedding.isEmpty()) {
            return SwapRunResult(
                statusMessage = "Source embedding is missing. ${sourceAnalysis.statusMessage}",
                outputBitmap = postprocessor.createPreviewBitmap(request.targetBitmap),
            )
        }

        // ── Multi-face mode: iterate over every matching face and chain results. ──────────
        if (request.faceFilterMode != FaceFilterMode.SPECIFIC) {
            val facesToSwap = when (request.faceFilterMode) {
                FaceFilterMode.ALL_FACES  -> targetAnalysis.allDetectedFaces
                FaceFilterMode.MALE_ONLY  -> targetAnalysis.allDetectedFaces.filter { it.isMale == true }
                FaceFilterMode.FEMALE_ONLY -> targetAnalysis.allDetectedFaces.filter { it.isMale == false }
                FaceFilterMode.SPECIFIC   -> emptyList() // unreachable
            }
            if (facesToSwap.isEmpty()) {
                return SwapRunResult(
                    statusMessage = "No faces matched the '${request.faceFilterMode.displayName}' filter.",
                    outputBitmap = postprocessor.createPreviewBitmap(request.targetBitmap),
                )
            }
            var currentBitmap = request.targetBitmap
            for ((swapIdx, face) in facesToSwap.withIndex()) {
                onProgress("Swapping face ${swapIdx + 1} / ${facesToSwap.size}…")
                val singleResult = runSwap(
                    SwapRequest(
                        sourceBitmap    = request.sourceBitmap,
                        targetBitmap    = currentBitmap,
                        enhancerEnabled = request.enhancerEnabled,
                        // Use the face's original rank index from the initial analysis.
                        // Faces are sorted best-first; swapping changes only texture so
                        // the ranking stays stable across iterations.
                        targetFaceIndex = face.index,
                        faceFilterMode  = FaceFilterMode.SPECIFIC,
                    ),
                    onProgress,
                )
                currentBitmap = singleResult.outputBitmap ?: currentBitmap
            }
            return SwapRunResult(
                statusMessage = "Multi-face swap complete. ${facesToSwap.size} face(s) swapped.",
                outputBitmap = currentBitmap,
            )
        }
        // ─────────────────────────────────────────────────────────────────────────────────

        // When targetFaceIndex > 0, prefer the keypoints and bounding box of the selected face
        // over the primary-face values stored in FaceAnalysisSummary.
        val selectedTargetFace = targetAnalysis.allDetectedFaces.getOrNull(request.targetFaceIndex)
        val effectiveTargetFivePoints = selectedTargetFace?.fiveKeypoints ?: targetAnalysis.landmarkFivePoints
        val effectiveTargetBox = selectedTargetFace?.box ?: targetAnalysis.primaryFaceBox

        val affineMatrix = effectiveTargetFivePoints?.let {
            runCatching {
                FaceAlignmentOps.getSimilarityTransformMatrix(
                    landmarks = it,
                    targetPoints = FaceAlignmentOps.defaultTargetPoints128(),
                )
            }.getOrNull()
        }
        val inverseAffineMatrix = affineMatrix?.let { forward ->
            Matrix().apply { forward.invert(this) }
        }

        val placementBox = effectiveTargetBox?.let {
            expandToSquareBox(
                box = it,
                imageWidth = request.targetBitmap.width,
                imageHeight = request.targetBitmap.height,
                scale = 1.12f,
            )
        }
        val targetFaceBitmap = placementBox?.let { cropBitmap(request.targetBitmap, it) }
            ?: targetAnalysis.primaryFaceBitmap
            ?: request.targetBitmap
        val targetRollDegrees = targetAnalysis.rollDegrees
            ?.takeIf { kotlin.math.abs(it) in 3f..40f }
            ?: 0f
        val alignedTargetFaceBitmap = if (targetRollDegrees != 0f) {
            rotateBitmapKeepingSize(targetFaceBitmap, -targetRollDegrees)
        } else {
            targetFaceBitmap
        }

        val kpLog = effectiveTargetFivePoints?.let { kps ->
            (0 until 5).joinToString(" ") { i -> "(${kps[i*2].toInt()},${kps[i*2+1].toInt()})" }
        } ?: "null"
        val emapVal = getEmap(inswapperFile)
        Log.d("OFFLINEMORPH_SWAP", "kps=$kpLog affine=${affineMatrix != null} emap=${emapVal?.size ?: "null"} targetFaces=${targetAnalysis.detectedFaces} srcFaces=${sourceAnalysis.detectedFaces} detMode=${targetAnalysis.statusMessage.substringAfter("Decoder=").take(20)}")

        val preparedTarget = if (affineMatrix != null) {
            val tensor = FaceAlignmentOps.preprocessTargetTensor(request.targetBitmap, affineMatrix)
            PreparedSwapTarget(
                resizedBitmap = alignedTargetFaceBitmap,
                inputTensor = floatBufferToArray(tensor),
                inputSize = FaceAlignmentOps.ALIGNED_SIZE,
            )
        } else {
            preprocessor.prepareTarget(alignedTargetFaceBitmap)
        }
        val previewBitmap = postprocessor.createPreviewBitmap(request.targetBitmap)

        sessionFactory.createSession(inswapperFile).use { swapSession ->
            val targetInputName = if (swapSession.inputNames.contains("target")) {
                "target"
            } else {
                swapSession.inputNames.firstOrNull() ?: "target"
            }
            val sourceInputName = if (swapSession.inputNames.contains("source")) {
                "source"
            } else {
                swapSession.inputNames.drop(1).firstOrNull() ?: "source"
            }

            val expectedSourceLength = (swapSession.inputInfo[sourceInputName]?.info as? TensorInfo)
                ?.shape
                ?.lastOrNull()
                ?.toInt()
                ?.takeIf { it > 0 }

            // Project the source embedding through the inswapper's emap matrix.
            // Python: latent = embedding @ emap; latent /= norm(latent)
            val emap = getEmap(inswapperFile)

            // ── Diagnostics ──────────────────────────────────────────────────
            val embNorm = sqrt(sourceEmbedding.fold(0f) { acc, v -> acc + v * v })
            val embNanCount = sourceEmbedding.count { it.isNaN() }
            Log.d("OFFLINEMORPH_DIAG", "srcEmb len=${sourceEmbedding.size} norm=$embNorm nans=$embNanCount" +
                " first3=${sourceEmbedding.take(3).joinToString { "%.4f".format(it) }}")
            emap?.let { e ->
                val emapNorm = sqrt(e.fold(0f) { acc, v -> acc + v * v })
                val emapNanCount = e.count { it.isNaN() }
                Log.d("OFFLINEMORPH_DIAG", "emap size=${e.size} norm=$emapNorm nans=$emapNanCount" +
                    " first3=${e.take(3).joinToString { "%.6f".format(it) }}" +
                    " last3=${e.takeLast(3).joinToString { "%.6f".format(it) }}")
            }
            // ─────────────────────────────────────────────────────────────────

            onProgress("Projecting Latents...")
            val compatibleEmbedding = if (emap != null) {
                projectEmbeddingThroughEmap(
                    rawEmbedding = sourceEmbedding,
                    emap = emap,
                    emapCols = expectedSourceLength ?: 512,
                ).also { latent ->
                    val latentNorm = sqrt(latent.fold(0f) { acc, v -> acc + v * v })
                    val latentNans = latent.count { it.isNaN() }
                    Log.d("OFFLINEMORPH_DIAG", "latent norm=$latentNorm nans=$latentNans" +
                        " first3=${latent.take(3).joinToString { "%.4f".format(it) }}")
                }
            } else {
                prepareCompatibleEmbedding(
                    rawEmbedding = sourceEmbedding,
                    expectedLength = expectedSourceLength,
                )
            }

            val swapRunMessage = runCatching {
                val targetShape = longArrayOf(
                    1,
                    3,
                    preparedTarget.inputSize.toLong(),
                    preparedTarget.inputSize.toLong(),
                )
                val sourceShape = longArrayOf(1, compatibleEmbedding.size.toLong())

                sessionFactory.createFloatTensor(preparedTarget.inputTensor, targetShape)
                    .use { targetTensor ->
                        sessionFactory.createFloatTensor(compatibleEmbedding, sourceShape)
                            .use { sourceTensor ->
                                onProgress("Running Inswapper...")
                                swapSession.run(
                                    mapOf(
                                        targetInputName to targetTensor,
                                        sourceInputName to sourceTensor,
                                    )
                                ).use { result ->
                                    val outputs = OrtValueUtils.extractFloatOutputs(result)
                                    val selectedOutput = selectSwapOutput(outputs, preparedTarget.inputSize)
                                    selectedOutput?.let { out ->
                                        val nanCount = out.data.count { it.isNaN() }
                                        val outMin = out.data.minOrNull() ?: Float.NaN
                                        val outMax = out.data.maxOrNull() ?: Float.NaN
                                        Log.d("OFFLINEMORPH_DIAG", "swap output: size=${out.data.size} nans=$nanCount min=$outMin max=$outMax")
                                    }
                                    val outputBitmap = selectedOutput?.let {
                                        postprocessor.toBitmapFromTensor(
                                            output = it,
                                            expectedSize = preparedTarget.inputSize,
                                        )
                                    }
                                    val affineComposed = if (selectedOutput != null && inverseAffineMatrix != null) {
                                        val chw = toChwTensor(selectedOutput, preparedTarget.inputSize)
                                        chw?.let { chwTensor ->
                                            val upscaledPatch = if (request.enhancerEnabled) {
                                                onProgress("Enhancing Facial Details...")
                                                enhancer.enhanceToBitmap(chwTensor)
                                            } else {
                                                null
                                            }
                                            onProgress("Blending Output...")
                                            if (upscaledPatch != null) {
                                                FaceAlignmentOps.compositeSwapBack(
                                                    swappedPatch = upscaledPatch,
                                                    originalTarget = request.targetBitmap,
                                                    inverseMatrix = inverseAffineMatrix,
                                                )
                                            } else {
                                                FaceAlignmentOps.compositeSwapBack(
                                                    swappedFaceTensor = chwTensor,
                                                    originalTarget = request.targetBitmap,
                                                    inverseMatrix = inverseAffineMatrix,
                                                )
                                            }
                                        }
                                    } else {
                                        null
                                    }

                                    if (affineComposed != null) {
                                        Log.d("OFFLINEMORPH_SWAP", "PATH=affine_composite outName=${selectedOutput?.name}")
                                        return SwapRunResult(
                                            statusMessage = "Real inswapper run succeeded with affine placement using output ${selectedOutput?.name ?: "unknown"}.",
                                            outputBitmap = affineComposed,
                                        )
                                    }

                                    if (outputBitmap != null) {
                                        val restoredOrientationFace = if (targetRollDegrees != 0f) {
                                            rotateBitmapKeepingSize(outputBitmap, targetRollDegrees)
                                        } else {
                                            outputBitmap
                                        }
                                        val composed = placementBox?.let {
                                            composeIntoTarget(
                                                target = request.targetBitmap,
                                                swappedFace = restoredOrientationFace,
                                                box = it,
                                            )
                                        } ?: restoredOrientationFace
                                        Log.d("OFFLINEMORPH_SWAP", "PATH=bbox_composite outName=${selectedOutput.name} tensorSize=${selectedOutput.data.size}")
                                        return SwapRunResult(
                                            statusMessage = "Real inswapper run succeeded with output ${selectedOutput.name} and tensor size ${selectedOutput.data.size}.",
                                            outputBitmap = composed,
                                        )
                                    }
                                }
                            }
                    }
                "Inswapper run executed, but output tensor shape did not match expected image layout."
            }.getOrElse { error ->
                Log.e("OFFLINEMORPH_SWAP", "Inswapper run failed", error)
                "Inswapper run failed: ${error.message}"
            }

            return SwapRunResult(
                statusMessage = buildString {
                    append("Source analysis: faces=")
                    append(sourceAnalysis.detectedFaces)
                    append(". ")
                    append(sourceAnalysis.statusMessage)
                    append(" Recognizer output=")
                    append(sourceAnalysis.recognizerOutputName ?: "unknown")
                    append(" embeddingLen=")
                    append(sourceAnalysis.embeddingLength)
                    append(" landmarkModel=")
                    append(sourceAnalysis.landmarkModelUsed ?: "none")
                    append(" landmarkOutput=")
                    append(sourceAnalysis.landmarkOutputName ?: "unknown")
                    append(" landmarkPoints=")
                    append(sourceAnalysis.landmarkPointCount)
                    append(" Target analysis: faces=")
                    append(targetAnalysis.detectedFaces)
                    append(". ")
                    append(targetAnalysis.statusMessage)
                    append(" targetRoll=")
                    append(targetRollDegrees)
                    append(" affine=")
                    append(if (affineMatrix != null) "enabled" else "fallback")
                    append(" Swap session ready with inputs ")
                    append(swapSession.inputNames.joinToString())
                    append(" and outputs ")
                    append(swapSession.outputNames.joinToString())
                    append(". Prepared target tensor for ")
                    append(preparedTarget.inputSize)
                    append("x")
                    append(preparedTarget.inputSize)
                    append(" inference. Source embedding length=")
                    append(sourceEmbedding.size)
                    append(" -> compatible=")
                    append(compatibleEmbedding.size)
                    append(". ")
                    append(swapRunMessage)
                },
                outputBitmap = previewBitmap,
            )
        }
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

    private fun expandToSquareBox(box: FaceBoundingBox, imageWidth: Int, imageHeight: Int, scale: Float): FaceBoundingBox {
        val cx = (box.left + box.right) / 2.0f
        val cy = (box.top + box.bottom) / 2.0f
        val side = max(box.right - box.left, box.bottom - box.top).toFloat() * scale
        val half = side / 2.0f

        val left = max(0, (cx - half).toInt())
        val top = max(0, (cy - half).toInt())
        val right = min(imageWidth, (cx + half).toInt())
        val bottom = min(imageHeight, (cy + half).toInt())

        return FaceBoundingBox(
            left = left,
            top = top,
            right = right,
            bottom = bottom,
            score = box.score,
        )
    }

    private fun cropBitmap(bitmap: Bitmap, box: FaceBoundingBox): Bitmap? {
        return runCatching {
            val left = box.left.coerceIn(0, bitmap.width - 1)
            val top = box.top.coerceIn(0, bitmap.height - 1)
            val right = box.right.coerceIn(left + 1, bitmap.width)
            val bottom = box.bottom.coerceIn(top + 1, bitmap.height)
            Bitmap.createBitmap(bitmap, left, top, right - left, bottom - top)
        }.getOrNull()
    }

    private fun composeIntoTarget(target: Bitmap, swappedFace: Bitmap, box: FaceBoundingBox): Bitmap {
        val output = target.copy(Bitmap.Config.ARGB_8888, true)
        val width = max(1, box.right - box.left)
        val height = max(1, box.bottom - box.top)
        val resizedFace = Bitmap.createScaledBitmap(swappedFace, width, height, true)

        val facePixels = IntArray(width * height)
        val targetPixels = IntArray(width * height)
        resizedFace.getPixels(facePixels, 0, width, 0, 0, width, height)
        output.getPixels(targetPixels, 0, width, box.left, box.top, width, height)

        val outPixels = IntArray(width * height)
        val blendStrength = 0.95f
        val edgeFeather = (min(width, height) * 0.14f).coerceAtLeast(4f)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val idx = y * width + x
                val distLeft = x.toFloat()
                val distRight = (width - 1 - x).toFloat()
                val distTop = y.toFloat()
                val distBottom = (height - 1 - y).toFloat()
                val edgeDistance = min(min(distLeft, distRight), min(distTop, distBottom))
                val edgeAlpha = (edgeDistance / edgeFeather).coerceIn(0f, 1f)
                val alpha = edgeAlpha * blendStrength

                val src = facePixels[idx]
                val dst = targetPixels[idx]
                val sr = (src shr 16) and 0xFF
                val sg = (src shr 8) and 0xFF
                val sb = src and 0xFF
                val dr = (dst shr 16) and 0xFF
                val dg = (dst shr 8) and 0xFF
                val db = dst and 0xFF

                val r = (dr * (1f - alpha) + sr * alpha).toInt().coerceIn(0, 255)
                val g = (dg * (1f - alpha) + sg * alpha).toInt().coerceIn(0, 255)
                val b = (db * (1f - alpha) + sb * alpha).toInt().coerceIn(0, 255)
                outPixels[idx] = (255 shl 24) or (r shl 16) or (g shl 8) or b
            }
        }

        output.setPixels(outPixels, 0, width, box.left, box.top, width, height)
        return output
    }

    private fun rotateBitmapKeepingSize(bitmap: Bitmap, degrees: Float): Bitmap {
        val out = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        val matrix = Matrix().apply {
            postTranslate(-bitmap.width / 2f, -bitmap.height / 2f)
            postRotate(degrees)
            postTranslate(bitmap.width / 2f, bitmap.height / 2f)
        }
        canvas.drawBitmap(bitmap, matrix, null)
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

    private fun floatBufferToArray(buffer: FloatBuffer): FloatArray {
        val dup = buffer.duplicate()
        dup.position(0)
        val out = FloatArray(dup.remaining())
        dup.get(out)
        return out
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

}
