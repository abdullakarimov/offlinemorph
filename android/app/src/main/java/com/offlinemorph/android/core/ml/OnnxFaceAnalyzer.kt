package com.offlinemorph.android.core.ml

import android.graphics.Bitmap
import ai.onnxruntime.OrtSession
import com.offlinemorph.android.feature.models.ModelCatalog
import java.io.File
import kotlin.math.atan2
import kotlin.math.max
import kotlin.math.min
import kotlin.math.PI
import kotlin.math.sqrt

class OnnxFaceAnalyzer(
    private val modelsDirectory: File,
    private val sessionFactory: OrtSessionFactory,
    private val geometryCache: FaceGeometryCache = FaceGeometryCache(),
) : FaceAnalyzer {
    private val squareCropPreprocessor = SquareCropPreprocessor()
    private val tensorConverter = BitmapTensorConverter()

    override suspend fun analyze(bitmap: Bitmap): FaceAnalysisSummary {
        geometryCache.get(bitmap)?.let { return it }
        val detectorFile = File(modelsDirectory, ModelCatalog.DETECTOR)
        val recognizerFile = File(modelsDirectory, ModelCatalog.RECOGNIZER)

        if (!detectorFile.isFile || !recognizerFile.isFile) {
            return FaceAnalysisSummary(
                detectedFaces = 0,
                statusMessage = "Face analysis models are not fully installed.",
            )
        }

        sessionFactory.createSession(detectorFile).use { detectorSession ->
            sessionFactory.createSession(recognizerFile).use { recognizerSession ->
                val detectorInput = squareCropPreprocessor.prepare(bitmap, 640)
                // SCRFD detector expects BGR CHW normalized as (pixel − 127.5) / 128.0.
                val detectorTensor = tensorConverter.toNormalizedChwFloatArray(
                    bitmap = detectorInput.bitmap,
                    mean = floatArrayOf(127.5f / 128f, 127.5f / 128f, 127.5f / 128f),
                    scaleDivisor = 128.0f,
                    swapChannels = true,
                )

                val detectorProbe = runSessionProbe(
                    session = detectorSession,
                    inputTensor = detectorTensor,
                    inputSize = detectorInput.bitmap.width,
                    label = "detector",
                )
                val detectorDecode = decodeDetectorOutputs(
                    detectorOutputs = detectorProbe.outputs,
                    fallbackOutput = detectorProbe.firstOutput,
                    detectorInputSize = detectorInput.bitmap.width,
                    detectorScale = detectorInput.scale,
                    sourceWidth = bitmap.width,
                    sourceHeight = bitmap.height,
                )
                val primaryFaceBox = detectorDecode.primaryFaceBox
                val primaryFaceBitmap = primaryFaceBox?.let { cropBitmap(bitmap, it) }

                val recognizerSource = primaryFaceBitmap ?: bitmap
                val recognizerInput = squareCropPreprocessor.prepare(recognizerSource, 112)
                // ArcFace / InsightFace w600k_r50 expects RGB CHW normalised as (pixel/255 − 0.5) / 0.5 = pixel/127.5 − 1.
                val recognizerTensor = tensorConverter.toNormalizedChwFloatArray(
                    bitmap = recognizerInput.bitmap,
                    mean = floatArrayOf(0.5f, 0.5f, 0.5f),
                    std = floatArrayOf(0.5f, 0.5f, 0.5f),
                    scaleDivisor = 255.0f,
                    swapChannels = false,
                )
                val recognizerProbe = runSessionProbe(
                    session = recognizerSession,
                    inputTensor = recognizerTensor,
                    inputSize = recognizerInput.bitmap.width,
                    label = "recognizer",
                )
                val landmarkProbe = runLandmarkProbe(
                    sourceFaceBitmap = recognizerSource,
                    sourceFaceBoxInOriginal = primaryFaceBox,
                )
                val estimatedFaceCount = detectorDecode.faceCount
                val embedding = extractEmbedding(recognizerProbe)

                // Build the per-face list from the detector's sorted NMS survivors.
                val rawDetectedFaces: List<DetectedFaceResult> = detectorDecode.allFaces
                    .mapIndexed { idx, (box, kps) ->
                        val thumb = cropBitmap(bitmap, box)
                        DetectedFaceResult(
                            index = idx,
                            box = box,
                            fiveKeypoints = kps,
                            thumbnail = thumb,
                        )
                    }

                // Optional gender classification via genderage.onnx.
                val genderageFile = File(modelsDirectory, ModelCatalog.GENDERAGE)
                val allDetectedFaces: List<DetectedFaceResult> = if (genderageFile.isFile && rawDetectedFaces.isNotEmpty()) {
                    sessionFactory.createSession(genderageFile).use { genderSession ->
                        rawDetectedFaces.map { face ->
                            val isMale = face.thumbnail?.let { classifyGender(it, genderSession) }
                            face.copy(isMale = isMale)
                        }
                    }
                } else {
                    rawDetectedFaces
                }

                return FaceAnalysisSummary(
                    detectedFaces = estimatedFaceCount,
                    statusMessage = "Face-analysis sessions loaded. Detector crop scale=${detectorInput.scale}. Decoder=${detectorDecode.mode}. Primary face=${primaryFaceBox?.let { "(${it.left},${it.top})-(${it.right},${it.bottom}) score=${it.score}" } ?: "not decoded"}. Recognizer crop scale=${recognizerInput.scale}. ${detectorProbe.message} ${recognizerProbe.message} ${landmarkProbe.message}",
                    embedding = embedding,
                    embeddingLength = embedding?.size ?: 0,
                    recognizerOutputName = recognizerProbe.selectedOutput?.name,
                    landmarkModelUsed = landmarkProbe.modelName,
                    landmarkOutputName = landmarkProbe.outputName,
                    landmarkPointCount = landmarkProbe.pointCount,
                    // Prefer the detector's native 5-point keypoints over the heuristic
                    // 106-point→5-point derivation; the detector kps are more accurate.
                    landmarkFivePoints = detectorDecode.detectorFivePoints ?: landmarkProbe.fivePoints,
                    leftEye = landmarkProbe.leftEye,
                    rightEye = landmarkProbe.rightEye,
                    rollDegrees = landmarkProbe.rollDegrees,
                    primaryFaceBox = primaryFaceBox,
                    primaryFaceBitmap = primaryFaceBitmap,
                    allDetectedFaces = allDetectedFaces,
                ).also { geometryCache.put(bitmap, it) }
            }
        }
    }

    private fun runSessionProbe(
        session: OrtSession,
        inputTensor: FloatArray,
        inputSize: Int,
        label: String,
    ): SessionProbeResult {
        return runCatching {
            val inputName = session.inputNames.firstOrNull()
                ?: error("$label session has no inputs")
            val shape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())

            sessionFactory.createFloatTensor(inputTensor, shape).use { tensor ->
                session.run(mapOf(inputName to tensor)).use { result ->
                    val outputSummaries = mutableListOf<String>()
                    for (entry in result) {
                        outputSummaries += "${entry.key}:${entry.value.info}"
                    }
                    val allOutputs = OrtValueUtils.extractFloatOutputs(result)
                    val selected = selectOutputForLabel(label, allOutputs)
                    val firstOutput = selected?.data ?: OrtValueUtils.extractFirstFloatArray(result)
                    SessionProbeResult(
                        ok = true,
                        message = "$label probe ok. input=$inputName shape=${shape.joinToString(prefix = "[", postfix = "]")}, outputs=${outputSummaries.joinToString(separator = " | ")}, selected=${selected?.name ?: "none"}, firstOutputSize=${firstOutput?.size ?: 0}",
                        firstOutput = firstOutput,
                        selectedOutput = selected,
                        outputs = allOutputs,
                    )
                }
            }
        }.getOrElse { error ->
            SessionProbeResult(
                ok = false,
                message = "$label probe failed: ${error.message}",
                firstOutput = null,
                selectedOutput = null,
                outputs = emptyList(),
            )
        }
    }

    private fun runLandmarkProbe(
        sourceFaceBitmap: Bitmap,
        sourceFaceBoxInOriginal: FaceBoundingBox?,
    ): LandmarkProbeSummary {
        val landmarkFile = chooseLandmarkModelFile() ?: return LandmarkProbeSummary(
            modelName = null,
            outputName = null,
            pointCount = 0,
            fivePoints = null,
            leftEye = null,
            rightEye = null,
            rollDegrees = null,
            message = "Landmark probe skipped: no landmark model file found.",
        )

        return runCatching {
            sessionFactory.createSession(landmarkFile).use { session ->
                val inputSize = inferSquareInputSize(session).coerceIn(96, 512)
                val prep = squareCropPreprocessor.prepare(sourceFaceBitmap, inputSize)
                val tensor = tensorConverter.toNormalizedChwFloatArray(prep.bitmap)
                val probe = runSessionProbe(
                    session = session,
                    inputTensor = tensor,
                    inputSize = prep.bitmap.width,
                    label = "landmark",
                )

                val selected = selectLandmarkOutput(probe.outputs)
                val fallback = probe.firstOutput
                val selectedData = selected?.data ?: fallback
                val pointCount = selectedData?.let { inferLandmarkPointCount(it) } ?: 0
                val rangeLabel = selectedData?.let { inferLandmarkRangeLabel(it, inputSize) } ?: "unknown"
                val pointsOriginal = selectedData?.let {
                    decodeLandmarkPointsToOriginal(
                        values = it,
                        rangeLabel = rangeLabel,
                        inputSize = inputSize,
                        prepScale = prep.scale,
                        sourceFaceBoxInOriginal = sourceFaceBoxInOriginal,
                    )
                }.orEmpty()
                val eyePair = estimateEyePair(pointsOriginal)
                val roll = eyePair?.let { estimateRollDegrees(it.first, it.second) }
                val fivePoints = deriveFivePointLandmarks(pointsOriginal, eyePair)

                LandmarkProbeSummary(
                    modelName = landmarkFile.name,
                    outputName = selected?.name,
                    pointCount = pointCount,
                    fivePoints = fivePoints,
                    leftEye = eyePair?.first,
                    rightEye = eyePair?.second,
                    rollDegrees = roll,
                    message = "Landmark probe model=${landmarkFile.name}, output=${selected?.name ?: "fallback"}, points=$pointCount, range=$rangeLabel, roll=${roll?.let { "%.1f".format(it) } ?: "unknown"}.",
                )
            }
        }.getOrElse { error ->
            LandmarkProbeSummary(
                modelName = landmarkFile.name,
                outputName = null,
                pointCount = 0,
                fivePoints = null,
                leftEye = null,
                rightEye = null,
                rollDegrees = null,
                message = "Landmark probe failed (${landmarkFile.name}): ${error.message}",
            )
        }
    }

    private fun deriveFivePointLandmarks(
        points: List<FacePoint>,
        eyePair: Pair<FacePoint, FacePoint>?,
    ): FloatArray? {
        if (points.isEmpty()) return null
        val leftEye = eyePair?.first ?: return null
        val rightEye = eyePair.second

        var minX = Float.POSITIVE_INFINITY
        var minY = Float.POSITIVE_INFINITY
        var maxX = Float.NEGATIVE_INFINITY
        var maxY = Float.NEGATIVE_INFINITY
        for (p in points) {
            if (p.x < minX) minX = p.x
            if (p.y < minY) minY = p.y
            if (p.x > maxX) maxX = p.x
            if (p.y > maxY) maxY = p.y
        }

        val width = (maxX - minX).coerceAtLeast(1f)
        val height = (maxY - minY).coerceAtLeast(1f)

        val noseCandidates = points.filter {
            it.x in (minX + width * 0.30f)..(minX + width * 0.70f) &&
                it.y in (minY + height * 0.35f)..(minY + height * 0.80f)
        }
        val nose = if (noseCandidates.isNotEmpty()) centroid(noseCandidates) else FacePoint(minX + width * 0.5f, minY + height * 0.60f)

        val mouthCandidates = points.filter { it.y >= minY + height * 0.62f }
        val mouthLeft = mouthCandidates.minByOrNull { it.x } ?: FacePoint(minX + width * 0.38f, minY + height * 0.80f)
        val mouthRight = mouthCandidates.maxByOrNull { it.x } ?: FacePoint(minX + width * 0.62f, minY + height * 0.80f)

        return floatArrayOf(
            leftEye.x, leftEye.y,
            rightEye.x, rightEye.y,
            nose.x, nose.y,
            mouthLeft.x, mouthLeft.y,
            mouthRight.x, mouthRight.y,
        )
    }

    private fun decodeLandmarkPointsToOriginal(
        values: FloatArray,
        rangeLabel: String,
        inputSize: Int,
        prepScale: Float,
        sourceFaceBoxInOriginal: FaceBoundingBox?,
    ): List<FacePoint> {
        val count = inferLandmarkPointCount(values)
        if (count <= 0) {
            return emptyList()
        }

        val dims = if (values.size >= count * 2) 2 else 3
        val out = ArrayList<FacePoint>(count)
        val offsetX = sourceFaceBoxInOriginal?.left?.toFloat() ?: 0f
        val offsetY = sourceFaceBoxInOriginal?.top?.toFloat() ?: 0f

        for (i in 0 until count) {
            val base = i * dims
            val rawX = values[base]
            val rawY = values[base + 1]
            val px = when (rangeLabel) {
                "normalized[-1,1]" -> ((rawX + 1f) * 0.5f) * inputSize
                "normalized[0,1]" -> rawX * inputSize
                else -> rawX
            }
            val py = when (rangeLabel) {
                "normalized[-1,1]" -> ((rawY + 1f) * 0.5f) * inputSize
                "normalized[0,1]" -> rawY * inputSize
                else -> rawY
            }

            val srcX = px / prepScale
            val srcY = py / prepScale
            out += FacePoint(x = offsetX + srcX, y = offsetY + srcY)
        }

        return out
    }

    private fun estimateEyePair(points: List<FacePoint>): Pair<FacePoint, FacePoint>? {
        if (points.size < 20) {
            return null
        }

        var minX = Float.POSITIVE_INFINITY
        var minY = Float.POSITIVE_INFINITY
        var maxX = Float.NEGATIVE_INFINITY
        var maxY = Float.NEGATIVE_INFINITY
        for (p in points) {
            if (p.x < minX) minX = p.x
            if (p.y < minY) minY = p.y
            if (p.x > maxX) maxX = p.x
            if (p.y > maxY) maxY = p.y
        }
        val width = (maxX - minX).coerceAtLeast(1f)
        val height = (maxY - minY).coerceAtLeast(1f)
        val midX = (minX + maxX) * 0.5f

        val eyeBandTop = minY + height * 0.18f
        val eyeBandBottom = minY + height * 0.62f

        val leftCandidates = points.filter { p ->
            p.x < midX && p.y in eyeBandTop..eyeBandBottom
        }
        val rightCandidates = points.filter { p ->
            p.x >= midX && p.y in eyeBandTop..eyeBandBottom
        }

        if (leftCandidates.size < 3 || rightCandidates.size < 3) {
            return null
        }

        val leftEye = centroid(leftCandidates)
        val rightEye = centroid(rightCandidates)
        if (rightEye.x <= leftEye.x) {
            return null
        }
        return leftEye to rightEye
    }

    private fun centroid(points: List<FacePoint>): FacePoint {
        var sx = 0f
        var sy = 0f
        for (p in points) {
            sx += p.x
            sy += p.y
        }
        val n = points.size.coerceAtLeast(1).toFloat()
        return FacePoint(sx / n, sy / n)
    }

    private fun estimateRollDegrees(leftEye: FacePoint, rightEye: FacePoint): Float {
        val dy = rightEye.y - leftEye.y
        val dx = (rightEye.x - leftEye.x).coerceAtLeast(1e-4f)
        return (atan2(dy, dx) * 180.0 / PI).toFloat()
    }

    private fun chooseLandmarkModelFile(): File? {
        val candidates = listOf(
            File(modelsDirectory, ModelCatalog.LANDMARKS_2D),
            File(modelsDirectory, ModelCatalog.LANDMARKS_3D),
        )
        return candidates.firstOrNull { it.isFile }
    }

    private fun inferSquareInputSize(session: OrtSession): Int {
        val firstInputName = session.inputNames.firstOrNull() ?: return 192
        val shape = (session.inputInfo[firstInputName]?.info as? ai.onnxruntime.TensorInfo)?.shape ?: return 192
        val positiveDims = shape.map { it.toInt() }.filter { it > 0 }
        val tail = positiveDims.takeLast(2)
        return if (tail.size == 2) minOf(tail[0], tail[1]).coerceAtLeast(96) else 192
    }

    private fun selectLandmarkOutput(outputs: List<TensorData>): TensorData? {
        if (outputs.isEmpty()) {
            return null
        }

        return outputs.maxWithOrNull(
            compareByDescending<TensorData> { landmarkOutputScore(it) }
                .thenByDescending { it.data.size }
        )
    }

    private fun landmarkOutputScore(output: TensorData): Int {
        var score = 0
        val name = output.name
        if (name.contains("landmark", ignoreCase = true) || name.contains("lmk", ignoreCase = true)) {
            score += 60
        }

        val points = inferLandmarkPointCount(output.data)
        score += when {
            points in 60..120 -> 100
            points in 20..200 -> 50
            else -> 0
        }

        val shape = output.shape
        if (shape != null && shape.size >= 2) {
            val last = shape.last().toInt()
            val penultimate = shape[shape.size - 2].toInt()
            if (last in listOf(2, 3) && penultimate in 20..200) {
                score += 60
            }
        }
        return score
    }

    private fun inferLandmarkPointCount(data: FloatArray): Int {
        return when {
            data.size % 2 == 0 -> data.size / 2
            data.size % 3 == 0 -> data.size / 3
            else -> 0
        }
    }

    private fun inferLandmarkRangeLabel(data: FloatArray, inputSize: Int): String {
        if (data.isEmpty()) {
            return "empty"
        }

        var minValue = Float.POSITIVE_INFINITY
        var maxValue = Float.NEGATIVE_INFINITY
        for (v in data) {
            if (v < minValue) minValue = v
            if (v > maxValue) maxValue = v
        }

        return when {
            minValue >= -1.5f && maxValue <= 1.5f -> "normalized[-1,1]"
            minValue >= -0.25f && maxValue <= 1.25f -> "normalized[0,1]"
            minValue >= -4f && maxValue <= inputSize + 4f -> "pixel-like"
            else -> "wide"
        }
    }

    private fun selectOutputForLabel(label: String, outputs: List<TensorData>): TensorData? {
        if (outputs.isEmpty()) {
            return null
        }

        return when (label) {
            "detector" -> outputs.maxByOrNull { output -> detectionConfidenceScore(output.data) }
            "recognizer" -> outputs.maxWithOrNull(
                compareByDescending<TensorData> { recognizerVectorScore(it) }
                    .thenByDescending { it.data.size }
            )
            else -> outputs.firstOrNull()
        }
    }

    private fun recognizerVectorScore(output: TensorData): Int {
        val nameScore = when {
            output.name.contains("fc1", ignoreCase = true) -> 80
            output.name.contains("embed", ignoreCase = true) -> 70
            output.name.contains("norm", ignoreCase = true) -> 60
            else -> 0
        }

        val shape = output.shape
        val lengthHint = when {
            shape == null || shape.isEmpty() -> output.data.size
            shape.size == 2 -> shape[1].toInt().coerceAtLeast(0)
            else -> shape.last().toInt().coerceAtLeast(0)
        }

        val shapeScore = when {
            lengthHint == 512 -> 100
            lengthHint in 256..1024 -> 60
            output.data.size in 256..2048 -> 40
            else -> 0
        }

        return nameScore + shapeScore
    }

    private fun extractEmbedding(probe: SessionProbeResult): FloatArray? {
        val raw = probe.selectedOutput?.data ?: probe.firstOutput ?: return null
        if (raw.isEmpty()) {
            return null
        }

        val targetLength = when {
            raw.size >= 512 -> 512
            raw.size >= 256 -> 256
            else -> raw.size
        }
        val trimmed = if (raw.size == targetLength) raw else raw.copyOf(targetLength)
        return l2Normalize(trimmed)
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

    private fun detectionConfidenceScore(data: FloatArray): Float {
        val strides = listOf(15, 6, 5)
        var best = -1f
        for (stride in strides) {
            if (data.size % stride != 0) {
                continue
            }
            var i = 0
            while (i + 4 < data.size) {
                val score = data[i + 4]
                if (score > best) {
                    best = score
                }
                i += stride
            }
        }
        return best
    }

    private fun decodeDetectorOutputs(
        detectorOutputs: List<TensorData>,
        fallbackOutput: FloatArray?,
        detectorInputSize: Int,
        detectorScale: Float,
        sourceWidth: Int,
        sourceHeight: Int,
    ): DetectorDecodeResult {
        if (detectorScale <= 0f) {
            return DetectorDecodeResult(primaryFaceBox = null, faceCount = 0, mode = "invalid-scale")
        }

        // Try SCRFD anchor-based decoding first (handles det_10g.onnx split-output format).
        val fromScrfd = decodeScrfdOutputs(
            detectorOutputs,
            detectorInputSize,
            detectorScale,
            sourceWidth,
            sourceHeight,
        )
        if (fromScrfd.primaryFaceBox != null) {
            return fromScrfd
        }

        val fromCombined = decodeCombinedOutputs(
            detectorOutputs,
            detectorInputSize,
            detectorScale,
            sourceWidth,
            sourceHeight,
        )
        if (fromCombined.primaryFaceBox != null) {
            return fromCombined.copy(mode = "combined-output")
        }

        val fromSplit = decodeSplitBoxScoreOutputs(
            detectorOutputs,
            detectorInputSize,
            detectorScale,
            sourceWidth,
            sourceHeight,
        )
        if (fromSplit.primaryFaceBox != null) {
            return fromSplit.copy(mode = "split-box-score")
        }

        val fromFallback = decodeFallbackOutput(
            fallbackOutput,
            detectorInputSize,
            detectorScale,
            sourceWidth,
            sourceHeight,
        )
        return fromFallback.copy(mode = "fallback-flat")
    }

    private fun decodeCombinedOutputs(
        detectorOutputs: List<TensorData>,
        detectorInputSize: Int,
        detectorScale: Float,
        sourceWidth: Int,
        sourceHeight: Int,
    ): DetectorDecodeResult {
        var best: FaceBoundingBox? = null
        var bestPriority = Float.NEGATIVE_INFINITY
        var bestRow = -1
        var bestData: FloatArray? = null
        var bestCols = 0
        var count = 0

        for (output in detectorOutputs) {
            val data = output.data
            val cols = inferColumns(output.shape, data.size)
            // Only process tensors with at least 15 columns (x1,y1,x2,y2,score,kp0x...kp4y).
            // Tensors with 10 cols are SCRFD raw-kps outputs and must NOT be treated as combined detections.
            if (cols < 15) {
                continue
            }

            val rows = data.size / cols
            for (row in 0 until rows) {
                val base = row * cols
                val score = data[base + 4]
                if (score <= 0.35f) {
                    continue
                }

                val candidate = mapBoundingBox(
                    x1 = data[base],
                    y1 = data[base + 1],
                    x2 = data[base + 2],
                    y2 = data[base + 3],
                    score = score,
                    detectorInputSize = detectorInputSize,
                    detectorScale = detectorScale,
                    sourceWidth = sourceWidth,
                    sourceHeight = sourceHeight,
                ) ?: continue

                count += 1
                val priority = primaryFacePriority(candidate, sourceWidth, sourceHeight)
                if (best == null || priority > bestPriority) {
                    best = candidate
                    bestPriority = priority
                    bestRow = row
                    bestData = data
                    bestCols = cols
                }
            }
        }

        val detectorKps = if (bestRow >= 0 && bestData != null && bestCols >= 15) {
            runCatching {
                decodeKeypointRow(bestData, bestRow, bestCols, kpOffset = 5, detectorInputSize, detectorScale)
            }.getOrNull()
        } else null

        return DetectorDecodeResult(best, count, mode = "combined", detectorFivePoints = detectorKps)
    }

    private fun decodeSplitBoxScoreOutputs(
        detectorOutputs: List<TensorData>,
        detectorInputSize: Int,
        detectorScale: Float,
        sourceWidth: Int,
        sourceHeight: Int,
    ): DetectorDecodeResult {
        val boxOutputs = detectorOutputs.filter { inferColumns(it.shape, it.data.size) == 4 }
        val scoreOutputs = detectorOutputs.filter { inferColumns(it.shape, it.data.size) in listOf(1, 2) }

        var best: FaceBoundingBox? = null
        var bestPriority = Float.NEGATIVE_INFINITY
        var bestRow = -1
        var bestBoxRows = 0
        var count = 0
        for (boxes in boxOutputs) {
            val boxRows = boxes.data.size / 4
            if (boxRows <= 0) continue

            val scoreTensor = scoreOutputs.firstOrNull { score ->
                val cols = inferColumns(score.shape, score.data.size)
                val scoreRows = score.data.size / cols
                scoreRows == boxRows
            } ?: continue

            val scoreCols = inferColumns(scoreTensor.shape, scoreTensor.data.size)

            for (row in 0 until boxRows) {
                val scoreBase = row * scoreCols
                val score = when (scoreCols) {
                    1 -> scoreTensor.data[scoreBase]
                    else -> scoreTensor.data[scoreBase + 1]
                }
                if (score <= 0.35f) continue

                val base = row * 4
                val candidate = mapBoundingBox(
                    x1 = boxes.data[base],
                    y1 = boxes.data[base + 1],
                    x2 = boxes.data[base + 2],
                    y2 = boxes.data[base + 3],
                    score = score,
                    detectorInputSize = detectorInputSize,
                    detectorScale = detectorScale,
                    sourceWidth = sourceWidth,
                    sourceHeight = sourceHeight,
                ) ?: continue

                count += 1
                val priority = primaryFacePriority(candidate, sourceWidth, sourceHeight)
                if (best == null || priority > bestPriority) {
                    best = candidate
                    bestPriority = priority
                    bestRow = row
                    bestBoxRows = boxRows
                }
            }
        }

        // Try to find a matching keypoint tensor (10 columns = 5 × (x, y)).
        val kpOutputs = detectorOutputs.filter { inferColumns(it.shape, it.data.size) == 10 }
        val detectorKps = if (best != null && bestRow >= 0 && bestBoxRows > 0) {
            val kpTensor = kpOutputs.firstOrNull { it.data.size / 10 == bestBoxRows }
            kpTensor?.let {
                runCatching {
                    decodeKeypointRow(it.data, bestRow, stride = 10, kpOffset = 0, detectorInputSize, detectorScale)
                }.getOrNull()
            }
        } else null

        return DetectorDecodeResult(best, count, mode = "split", detectorFivePoints = detectorKps)
    }

    private fun decodeFallbackOutput(
        detectorOutput: FloatArray?,
        detectorInputSize: Int,
        detectorScale: Float,
        sourceWidth: Int,
        sourceHeight: Int,
    ): DetectorDecodeResult {
        if (detectorOutput == null || detectorOutput.size < 5) {
            return DetectorDecodeResult(null, 0, mode = "fallback-empty")
        }

        val candidateStrides = listOf(15, 6, 5)
        var best: FaceBoundingBox? = null
        var bestPriority = Float.NEGATIVE_INFINITY
        var bestKps: FloatArray? = null
        var bestBaseOffset = -1
        var bestStride = 0
        var count = 0

        for (stride in candidateStrides) {
            if (detectorOutput.size % stride != 0) {
                continue
            }

            var i = 0
            while (i + 4 < detectorOutput.size) {
                var x1 = detectorOutput[i]
                var y1 = detectorOutput[i + 1]
                var x2 = detectorOutput[i + 2]
                var y2 = detectorOutput[i + 3]
                val score = detectorOutput[i + 4]

                if (score <= 0.35f) {
                    i += stride
                    continue
                }

                val candidate = mapBoundingBox(
                    x1 = x1,
                    y1 = y1,
                    x2 = x2,
                    y2 = y2,
                    score = score,
                    detectorInputSize = detectorInputSize,
                    detectorScale = detectorScale,
                    sourceWidth = sourceWidth,
                    sourceHeight = sourceHeight,
                )
                if (candidate == null) {
                    i += stride
                    continue
                }

                count += 1
                val priority = primaryFacePriority(candidate, sourceWidth, sourceHeight)
                if (best == null || priority > bestPriority) {
                    best = candidate
                    bestPriority = priority
                    bestBaseOffset = i
                    bestStride = stride
                }

                i += stride
            }

            if (best != null) {
                // Extract 5-point keypoints when stride==15 (x1,y1,x2,y2,score,kp1x,kp1y,...kp5y).
                if (bestStride == 15 && bestBaseOffset >= 0) {
                    bestKps = runCatching {
                        decodeKeypointRow(
                            data = detectorOutput,
                            rowIndex = bestBaseOffset / bestStride,
                            stride = bestStride,
                            kpOffset = 5,
                            detectorInputSize = detectorInputSize,
                            detectorScale = detectorScale,
                        )
                    }.getOrNull()
                }
                return DetectorDecodeResult(best, count, mode = "fallback-stride-$bestStride",
                    detectorFivePoints = bestKps)
            }
        }

        return DetectorDecodeResult(null, 0, mode = "fallback-none")
    }

    /**
     * Decodes 5 keypoints from one row of a detector output tensor.
     *
     * Mirrors Python: kps = kp_data[row] reshaped (5,2) * detectorInputSize / detectorScale
     *
     * @param data        flat float array (the entire tensor's data)
     * @param rowIndex    zero-based row index of the best detection
     * @param stride      number of columns per row in this tensor (e.g. 10 for pure-kp, 15 for combined)
     * @param kpOffset    index within the row where keypoints start (0 for pure-kp, 5 for combined)
     * @return FloatArray of size 10: [kp0x, kp0y, kp1x, kp1y, ..., kp4x, kp4y] in source-image coordinates
     */
    private fun decodeKeypointRow(
        data: FloatArray,
        rowIndex: Int,
        stride: Int,
        kpOffset: Int,
        detectorInputSize: Int,
        detectorScale: Float,
    ): FloatArray {
        val base = rowIndex * stride + kpOffset
        val out = FloatArray(10)
        for (i in 0 until 5) {
            var kpX = data[base + i * 2]
            var kpY = data[base + i * 2 + 1]
            // If in normalized [0..1] space, scale to pixel space first.
            if (kpX <= 1.5f && kpY <= 1.5f) {
                kpX *= detectorInputSize
                kpY *= detectorInputSize
            }
            out[i * 2] = kpX / detectorScale
            out[i * 2 + 1] = kpY / detectorScale
        }
        return out
    }

    private fun primaryFacePriority(box: FaceBoundingBox, imageWidth: Int, imageHeight: Int): Float {
        val width = (box.right - box.left).coerceAtLeast(1).toFloat()
        val height = (box.bottom - box.top).coerceAtLeast(1).toFloat()
        val areaNorm = ((width * height) / (imageWidth * imageHeight).toFloat()).coerceIn(0f, 1f)

        val cx = (box.left + box.right) * 0.5f
        val cy = (box.top + box.bottom) * 0.5f
        val dx = (cx - imageWidth * 0.5f) / imageWidth.coerceAtLeast(1)
        val dy = (cy - imageHeight * 0.5f) / imageHeight.coerceAtLeast(1)
        val centerDistance = sqrt(dx * dx + dy * dy)
        val centerScore = (1f - centerDistance / 0.7071f).coerceIn(0f, 1f)

        return box.score * 0.40f + areaNorm * 0.45f + centerScore * 0.15f
    }

    private fun inferColumns(shape: LongArray?, dataSize: Int): Int {
        if (shape == null || shape.isEmpty()) {
            return -1
        }
        val last = shape.last().toInt()
        if (last > 0 && dataSize % last == 0) {
            return last
        }
        return -1
    }

    private fun mapBoundingBox(
        x1: Float,
        y1: Float,
        x2: Float,
        y2: Float,
        score: Float,
        detectorInputSize: Int,
        detectorScale: Float,
        sourceWidth: Int,
        sourceHeight: Int,
    ): FaceBoundingBox? {
        var bx1 = x1
        var by1 = y1
        var bx2 = x2
        var by2 = y2

        if (bx2 <= 1.5f && by2 <= 1.5f) {
            bx1 *= detectorInputSize
            by1 *= detectorInputSize
            bx2 *= detectorInputSize
            by2 *= detectorInputSize
        }

        val mappedLeft = (bx1 / detectorScale).toInt().coerceIn(0, sourceWidth - 1)
        val mappedTop = (by1 / detectorScale).toInt().coerceIn(0, sourceHeight - 1)
        val mappedRight = (bx2 / detectorScale).toInt().coerceIn(0, sourceWidth)
        val mappedBottom = (by2 / detectorScale).toInt().coerceIn(0, sourceHeight)

        if (mappedRight - mappedLeft < 30 || mappedBottom - mappedTop < 30) {
            return null
        }

        return FaceBoundingBox(
            left = mappedLeft,
            top = mappedTop,
            right = mappedRight,
            bottom = mappedBottom,
            score = score,
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

    /**
     * Classifies the gender of a face crop using the genderage.onnx model.
     *
     * The model expects a 96×96 RGB crop normalised as (pixel/255 − 0.5) / 0.5.
     * Output[0] = female score, output[1] = male score (argmax of first two).
     *
     * Returns true (male), false (female), or null on any failure.
     */
    private fun classifyGender(faceThumbnail: Bitmap, session: ai.onnxruntime.OrtSession): Boolean? {
        return runCatching {
            val scaled = Bitmap.createScaledBitmap(faceThumbnail, 96, 96, true)
            val tensor = tensorConverter.toNormalizedChwFloatArray(
                bitmap = scaled,
                mean = floatArrayOf(0.5f, 0.5f, 0.5f),
                std = floatArrayOf(0.5f, 0.5f, 0.5f),
                scaleDivisor = 255.0f,
                swapChannels = false,
            )
            val shape = longArrayOf(1, 3, 96, 96)
            val inputName = session.inputNames.firstOrNull() ?: return null
            sessionFactory.createFloatTensor(tensor, shape).use { inputTensor ->
                session.run(mapOf(inputName to inputTensor)).use { result ->
                    val outputs = OrtValueUtils.extractFloatOutputs(result)
                    val out = outputs.firstOrNull()?.data ?: return null
                    if (out.size < 2) return null
                    // out[0] = female probability, out[1] = male probability
                    out[1] > out[0]
                }
            }
        }.getOrNull()
    }

    /**
     * Decodes SCRFD (e.g. det_10g.onnx) outputs.
     *
     * The model emits 9 separate tensors — one score (N,1), one bbox (N,4), and one kps (N,10)
     * per stride level (8, 16, 32).  The bbox values are distances from the anchor centre, and
     * the kps values are offsets from the anchor centre, following InsightFace conventions:
     *
     *   anchor_cx = (row // 2 % gridW) * stride
     *   anchor_cy = (row // 2 // gridW) * stride
     *   x1 = cx - d0,  y1 = cy - d1,  x2 = cx + d2,  y2 = cy + d3   (distance2bbox)
     *   kp_x = cx + offset_x,  kp_y = cy + offset_y                  (distance2kps)
     */
    private fun decodeScrfdOutputs(
        detectorOutputs: List<TensorData>,
        detectorInputSize: Int,
        detectorScale: Float,
        sourceWidth: Int,
        sourceHeight: Int,
    ): DetectorDecodeResult {
        val score1col = detectorOutputs.filter { inferColumns(it.shape, it.data.size) == 1 }
        val bbox4col  = detectorOutputs.filter { inferColumns(it.shape, it.data.size) == 4 }
        val kps10col  = detectorOutputs.filter { inferColumns(it.shape, it.data.size) == 10 }

        if (score1col.isEmpty() || bbox4col.isEmpty() || kps10col.isEmpty()) {
            return DetectorDecodeResult(null, 0, mode = "scrfd-missing")
        }

        // Sort each group descending by row count so strides 8 > 16 > 32 align at index 0,1,2.
        val scoresByRows = score1col.sortedByDescending { it.data.size }
        val bboxByRows  = bbox4col.sortedByDescending  { it.data.size / 4 }
        val kpsByRows   = kps10col.sortedByDescending  { it.data.size / 10 }

        val numStrides = minOf(scoresByRows.size, bboxByRows.size, kpsByRows.size)

        // Accumulate detections across all stride levels.
        val boxes  = mutableListOf<FaceBoundingBox>()
        val allKps = mutableListOf<FloatArray>()
        var totalPassed = 0

        for (si in 0 until numStrides) {
            val scoreData = scoresByRows[si]
            val bboxData  = bboxByRows[si]
            val kpsData   = kpsByRows[si]

            val numRows    = scoreData.data.size
            val numBboxRows = bboxData.data.size / 4
            val numKpsRows  = kpsData.data.size / 10
            if (numRows != numBboxRows || numRows != numKpsRows || numRows == 0) continue

            // gridW = sqrt(numRows / 2) — works for square grids with 2 anchors per cell.
            val gridW = sqrt(numRows / 2.0).toInt()
            if (gridW * gridW * 2 != numRows) continue
            val stride = detectorInputSize / gridW
            if (stride <= 0) continue

            for (row in 0 until numRows) {
                val score = scoreData.data[row]
                if (score <= 0.6f) continue
                totalPassed++

                // Anchor centre for this row.
                val gridPos  = row / 2
                val s = stride.toFloat()
                val anchorCx = (gridPos % gridW) * s
                val anchorCy = (gridPos / gridW) * s

                // distance2bbox: raw distances must be multiplied by stride (matches Python: bbox_preds *= stride).
                val x1 = anchorCx - bboxData.data[row * 4 + 0] * s
                val y1 = anchorCy - bboxData.data[row * 4 + 1] * s
                val x2 = anchorCx + bboxData.data[row * 4 + 2] * s
                val y2 = anchorCy + bboxData.data[row * 4 + 3] * s

                val box = mapBoundingBox(
                    x1 = x1, y1 = y1, x2 = x2, y2 = y2,
                    score = score,
                    detectorInputSize = detectorInputSize,
                    detectorScale = detectorScale,
                    sourceWidth = sourceWidth,
                    sourceHeight = sourceHeight,
                ) ?: continue

                // distance2kps: raw offsets must be multiplied by stride (matches Python: kps_preds *= stride).
                val kps = FloatArray(10)
                for (k in 0 until 5) {
                    kps[k * 2]     = (anchorCx + kpsData.data[row * 10 + k * 2]     * s) / detectorScale
                    kps[k * 2 + 1] = (anchorCy + kpsData.data[row * 10 + k * 2 + 1] * s) / detectorScale
                }

                boxes.add(box)
                allKps.add(kps)
            }
        }

        if (boxes.isEmpty()) {
            return DetectorDecodeResult(null, totalPassed, mode = "scrfd-no-box")
        }

        // Apply IoU NMS, then pick the best face by priority.
        val kept = scrfdNms(boxes, allKps, iouThreshold = 0.45f)

        // Sort all surviving faces best-first so index 0 == the primary face.
        val sortedFaces: List<Pair<FaceBoundingBox, FloatArray?>> = kept.sortedByDescending { (box, _) ->
            primaryFacePriority(box, sourceWidth, sourceHeight)
        }

        var bestBox: FaceBoundingBox? = null
        var bestKps: FloatArray?      = null
        var bestPriority = Float.NEGATIVE_INFINITY
        for ((box, kps) in kept) {
            val p = primaryFacePriority(box, sourceWidth, sourceHeight)
            if (p > bestPriority) { bestPriority = p; bestBox = box; bestKps = kps }
        }

        return DetectorDecodeResult(
            primaryFaceBox    = bestBox,
            faceCount         = kept.size,
            mode              = "scrfd-anchor",
            detectorFivePoints = bestKps,
            allFaces          = sortedFaces,
        )
    }

    /** Simple greedy NMS for SCRFD detections. */
    private fun scrfdNms(
        boxes: List<FaceBoundingBox>,
        kpsList: List<FloatArray>,
        iouThreshold: Float,
    ): List<Pair<FaceBoundingBox, FloatArray>> {
        val order = boxes.indices.sortedByDescending { boxes[it].score }
        val suppressed = BooleanArray(boxes.size)
        val result = mutableListOf<Pair<FaceBoundingBox, FloatArray>>()
        for (i in order) {
            if (suppressed[i]) continue
            result.add(boxes[i] to kpsList[i])
            val a = boxes[i]
            for (j in order) {
                if (j <= i || suppressed[j]) continue
                if (scrfdIou(a, boxes[j]) > iouThreshold) suppressed[j] = true
            }
        }
        return result
    }

    private fun scrfdIou(a: FaceBoundingBox, b: FaceBoundingBox): Float {
        val ix1 = max(a.left, b.left);   val iy1 = max(a.top, b.top)
        val ix2 = min(a.right, b.right); val iy2 = min(a.bottom, b.bottom)
        val iw = (ix2 - ix1).coerceAtLeast(0)
        val ih = (iy2 - iy1).coerceAtLeast(0)
        val inter = (iw * ih).toFloat()
        val aArea = ((a.right - a.left) * (a.bottom - a.top)).toFloat()
        val bArea = ((b.right - b.left) * (b.bottom - b.top)).toFloat()
        val union = aArea + bArea - inter
        return if (union <= 0f) 0f else inter / union
    }
}

private data class SessionProbeResult(
    val ok: Boolean,
    val message: String,
    val firstOutput: FloatArray?,
    val selectedOutput: TensorData?,
    val outputs: List<TensorData>,
)

private data class DetectorDecodeResult(
    val primaryFaceBox: FaceBoundingBox?,
    val faceCount: Int,
    val mode: String,
    val detectorFivePoints: FloatArray? = null,
    /** All NMS-surviving faces sorted best-first by priority score (index 0 == primary face). */
    val allFaces: List<Pair<FaceBoundingBox, FloatArray?>> = emptyList(),
)

private data class LandmarkProbeSummary(
    val modelName: String?,
    val outputName: String?,
    val pointCount: Int,
    val fivePoints: FloatArray?,
    val leftEye: FacePoint?,
    val rightEye: FacePoint?,
    val rollDegrees: Float?,
    val message: String,
)
