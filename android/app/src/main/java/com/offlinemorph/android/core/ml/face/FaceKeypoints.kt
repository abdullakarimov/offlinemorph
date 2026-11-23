package com.offlinemorph.android.core.ml.face

import com.offlinemorph.android.core.ml.FaceAnalysisSummary

/**
 * 5-point keypoints of up to [max] detected faces, best first. Detectors that only report the
 * primary face's landmarks contribute that one face.
 */
fun FaceAnalysisSummary.faceKeypoints(max: Int): List<FloatArray> =
    allDetectedFaces
        .mapNotNull { face -> face.fiveKeypoints ?: landmarkFivePoints?.takeIf { face.index == 0 } }
        .ifEmpty { listOfNotNull(landmarkFivePoints) }
        .take(max)
