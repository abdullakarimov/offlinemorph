package com.offlinemorph.android.core.ml.aging

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.TensorInfo
import android.graphics.Bitmap
import android.util.Log
import com.offlinemorph.android.core.ml.EngineError
import com.offlinemorph.android.core.ml.EngineProgress
import com.offlinemorph.android.core.ml.EngineResult
import com.offlinemorph.android.core.ml.FaceAlignmentOps
import com.offlinemorph.android.core.ml.FaceAnalyzer
import com.offlinemorph.android.core.ml.FaceTemplate
import com.offlinemorph.android.core.ml.OrtSessionFactory
import com.offlinemorph.android.core.ml.OrtValueUtils
import com.offlinemorph.android.feature.models.ModelCatalog
import java.io.File

/**
 * On-device aging / de-aging with FRAN (Face Re-Aging Network, Zoss et al. 2022).
 *
 * Model: [ModelCatalog.FRAN] — the MIT-licensed U-Net re-implementation
 * (timroelofs123/face_re-aging) as exported to ONNX by FaceFusion. The pipeline mirrors
 * FaceFusion's `age_modifier` for the `fran` model:
 *
 * 1. Align each face to the FFHQ 5-point template at 1024×1024.
 * 2. Feed `target` = RGB CHW in [0, 1] and `direction` = [current_age, target_age] / 100,
 *    both clipped to [0, 1].
 * 3. The output is a full 1024×1024 RGB image in [0, 1]; it is pasted back with a feathered
 *    box matte whose strength is [AgingRequest.intensity].
 *
 * The current age comes from genderage.onnx when installed (via [FaceAnalyzer] with
 * `includeGender = true`), else from [AgingRequest.currentAgeYears], else [DEFAULT_AGE].
 */
class OnDeviceAgingEngine(
    private val modelsDirectory: File,
    private val faceAnalyzer: FaceAnalyzer,
    private val sessionFactory: OrtSessionFactory = OrtSessionFactory(),
) : AgingEngine {

    override suspend fun runAging(
        request: AgingRequest,
        onProgress: (EngineProgress) -> Unit,
    ): EngineResult<AgingResult> {
        val modelFile = File(modelsDirectory, ModelCatalog.FRAN)
        if (!modelFile.isFile) {
            return EngineResult.Failure(
                error = EngineError.ModelNotFound(ModelCatalog.FRAN),
                statusMessage = "Aging model is not downloaded. Open Setup → Download All to get it.",
            )
        }
        val offset = request.ageOffsetYears.coerceIn(-MAX_OFFSET, MAX_OFFSET)
        val intensity = request.intensity.coerceIn(0f, 1f)

        return try {
            onProgress(EngineProgress("Detecting faces…", 0.05f))
            val analysis = faceAnalyzer.analyze(request.sourceBitmap, includeGender = true)
            // (keypoints, estimated age) per face. Detectors without per-face keypoints still
            // report the primary face's landmarks.
            val faces = analysis.allDetectedFaces
                .mapNotNull { face ->
                    val kps = face.fiveKeypoints ?: analysis.landmarkFivePoints?.takeIf { face.index == 0 }
                    kps?.let { it to face.age }
                }
                .ifEmpty { listOfNotNull(analysis.landmarkFivePoints?.let { it to null }) }
                .take(MAX_FACES)
            if (faces.isEmpty()) {
                return EngineResult.Failure(
                    error = EngineError.NoFaceDetected("aging"),
                    statusMessage = "No face found in the photo.",
                )
            }

            val session = sessionFactory.session(modelFile)
            var output = request.sourceBitmap
            for ((i, face) in faces.withIndex()) {
                onProgress(
                    EngineProgress(
                        if (faces.size > 1) "Re-aging face ${i + 1} / ${faces.size}…" else "Re-aging face…",
                        0.1f + 0.85f * i / faces.size,
                    ),
                )
                val (keypoints, estimatedAge) = face
                val currentAge = estimatedAge ?: request.currentAgeYears ?: DEFAULT_AGE
                val previous = output
                output = ageFace(session, output, keypoints, currentAge, offset, intensity)
                if (previous !== request.sourceBitmap && previous !== output) previous.recycle()
            }
            onProgress(EngineProgress("Done", 1f))
            EngineResult.Success(AgingResult(outputBitmap = output, appliedAgeOffset = offset))
        } catch (e: Exception) {
            Log.e(TAG, "FRAN aging failed", e)
            EngineResult.Failure(
                error = EngineError.InferenceFailure(e),
                statusMessage = e.message ?: "Aging inference failed.",
            )
        }
    }

    private fun ageFace(
        session: ai.onnxruntime.OrtSession,
        image: Bitmap,
        keypoints: FloatArray,
        currentAge: Int,
        offset: Int,
        intensity: Float,
    ): Bitmap {
        val toCrop = FaceAlignmentOps.alignmentMatrix(keypoints, FaceTemplate.FFHQ_512, SIZE)
        val crop = FaceAlignmentOps.warpToCrop(image, toCrop, SIZE)
        val input = FaceAlignmentOps.rgbMatToChw(crop, mean = 0f, std = 255f)
        crop.release()

        val direction = floatArrayOf(
            (currentAge / 100f).coerceIn(0f, 1f),
            ((currentAge + offset) / 100f).coerceIn(0f, 1f),
        )
        val plane = SIZE * SIZE
        val tensors = mutableMapOf<String, OnnxTensor>()
        val aged = try {
            for (name in session.inputNames) {
                tensors[name] = when {
                    name.contains("direction", ignoreCase = true) ->
                        sessionFactory.createFloatTensor(direction, directionShape(session, name))
                    else -> sessionFactory.createFloatTensor(input, longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong()))
                }
            }
            session.run(tensors).use { result ->
                OrtValueUtils.extractFloatOutputs(result).firstOrNull { it.data.size == plane * 3 }?.data
                    ?: error("FRAN returned no ${SIZE}×$SIZE RGB output")
            }
        } finally {
            tensors.values.forEach { it.close() }
        }
        for (i in aged.indices) aged[i] = aged[i].coerceIn(0f, 1f)

        val patch = FaceAlignmentOps.chwToRgbMat(aged, SIZE)
        try {
            return FaceAlignmentOps.pasteBack(
                target = image,
                patch = patch,
                patchToTarget = FaceAlignmentOps.invert(toCrop),
                seamless = false,
                strength = intensity,
            )
        } finally {
            patch.release()
        }
    }

    /** `direction` is declared as (2) in FaceFusion's export; honour (1, 2) too if a graph uses it. */
    private fun directionShape(session: ai.onnxruntime.OrtSession, name: String): LongArray {
        val declared = (session.inputInfo[name]?.info as? TensorInfo)?.shape
        return if (declared != null && declared.size == 2) longArrayOf(1, 2) else longArrayOf(2)
    }

    private companion object {
        const val TAG = "AgingEngine"
        const val SIZE = 1024
        const val MAX_OFFSET = 50
        const val MAX_FACES = 4
        /** Assumed current age when neither genderage.onnx nor the caller provides one. */
        const val DEFAULT_AGE = 30
    }
}
