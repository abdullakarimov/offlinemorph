package com.offlinemorph.android.core.ml

import android.graphics.Bitmap

data class FacePoint(
    val x: Float,
    val y: Float,
)

data class FaceBoundingBox(
    val left: Int,
    val top: Int,
    val right: Int,
    val bottom: Int,
    val score: Float,
)

/**
 * One detected face returned by the face analyzer.
 *
 * [index] is the position within [FaceAnalysisSummary.allDetectedFaces], sorted best-first by
 * the detector's priority score (confidence × size × center-prior).
 * [fiveKeypoints] is the 5-point landmark array [kp0x, kp0y, …, kp4x, kp4y] in source-image
 * coordinates, or null when keypoints are unavailable for this face.
 * [thumbnail] is a cropped bitmap of the face region, suitable for display in a selection UI.
 */
data class DetectedFaceResult(
    val index: Int,
    val box: FaceBoundingBox,
    val fiveKeypoints: FloatArray?,
    val thumbnail: Bitmap?,
    /** True = male, false = female, null = unknown (genderage.onnx not installed). */
    val isMale: Boolean? = null,
)

data class FaceAnalysisSummary(
    val detectedFaces: Int,
    val statusMessage: String,
    val embedding: FloatArray? = null,
    val embeddingLength: Int = 0,
    val recognizerOutputName: String? = null,
    val landmarkModelUsed: String? = null,
    val landmarkOutputName: String? = null,
    val landmarkPointCount: Int = 0,
    val landmarkFivePoints: FloatArray? = null,
    val leftEye: FacePoint? = null,
    val rightEye: FacePoint? = null,
    val rollDegrees: Float? = null,
    val primaryFaceBox: FaceBoundingBox? = null,
    val primaryFaceBitmap: Bitmap? = null,
    /** All faces detected and surviving NMS, sorted best-first by priority score. */
    val allDetectedFaces: List<DetectedFaceResult> = emptyList(),
)

interface FaceAnalyzer {
    suspend fun analyze(bitmap: Bitmap): FaceAnalysisSummary
}
