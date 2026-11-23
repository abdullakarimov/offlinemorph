package com.offlinemorph.android.core.video

import kotlin.math.hypot

/**
 * Temporal smoothing of 5-point face keypoints across video frames.
 *
 * Detector landmarks jitter by a pixel or two between frames even on a still face, and each
 * jitter shifts the aligned crop, which shows up as a shimmering swapped face. Each face is
 * tracked (matched to the nearest track from the previous frame) and its keypoints are blended
 * with an adaptive exponential moving average: nearly frozen when the face is still, and
 * following almost immediately when it moves, so smoothing never lags real motion.
 *
 * Not thread-safe; use one instance per video.
 */
class KeypointStabilizer(
    /** Blend weight of the new frame when the face is still (lower = smoother). */
    private val minAlpha: Float = 0.25f,
    /** Movement (in inter-ocular distances per frame) at which smoothing is fully off. */
    private val fullSpeed: Float = 0.12f,
) {
    private var tracks: List<FloatArray> = emptyList()

    fun reset() {
        tracks = emptyList()
    }

    /** Returns smoothed copies of [faces] (each 10 floats), in the same order. */
    fun stabilize(faces: List<FloatArray>): List<FloatArray> {
        val unused = tracks.toMutableList()
        val result = faces.map { kps ->
            val iod = eyeDistance(kps).coerceAtLeast(1f)
            val match = unused.minByOrNull { centreDistance(it, kps) }
                ?.takeIf { centreDistance(it, kps) < iod * 0.8f }
            if (match == null) {
                kps.copyOf()
            } else {
                unused.remove(match)
                val speed = meanPointDistance(match, kps) / iod
                val alpha = (minAlpha + (1f - minAlpha) * (speed / fullSpeed)).coerceIn(minAlpha, 1f)
                FloatArray(kps.size) { i -> match[i] + (kps[i] - match[i]) * alpha }
            }
        }
        tracks = result.map { it.copyOf() }
        return result
    }

    private fun eyeDistance(k: FloatArray) = hypot(k[2] - k[0], k[3] - k[1])

    private fun centreDistance(a: FloatArray, b: FloatArray): Float {
        var ax = 0f; var ay = 0f; var bx = 0f; var by = 0f
        for (i in 0 until 5) {
            ax += a[i * 2]; ay += a[i * 2 + 1]; bx += b[i * 2]; by += b[i * 2 + 1]
        }
        return hypot((ax - bx) / 5f, (ay - by) / 5f)
    }

    private fun meanPointDistance(a: FloatArray, b: FloatArray): Float {
        var sum = 0f
        for (i in 0 until 5) sum += hypot(a[i * 2] - b[i * 2], a[i * 2 + 1] - b[i * 2 + 1])
        return sum / 5f
    }
}
