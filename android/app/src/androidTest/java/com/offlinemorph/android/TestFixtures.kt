package com.offlinemorph.android

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import com.offlinemorph.android.core.ml.DetectedFaceResult
import com.offlinemorph.android.core.ml.FaceAnalysisSummary
import com.offlinemorph.android.core.ml.FaceAnalyzer
import com.offlinemorph.android.core.ml.FaceBoundingBox
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Deterministic synthetic inputs for instrumented tests. Nothing here depends on photos of real
 * people, so the test APK ships no personal data.
 */
object TestFixtures {

    /** Diagonal-gradient image with seeded noise: textured enough for filters and swaps to act on. */
    fun noiseBitmap(width: Int, height: Int, seed: Int = 1): Bitmap {
        val random = Random(seed)
        val pixels = IntArray(width * height) { i ->
            val x = i % width; val y = i / width
            val base = ((x + y) / 16) and 0xFF
            val n = random.nextInt(32)
            (0xFF shl 24) or (((base + n) and 0xFF) shl 16) or (((base / 2 + n) and 0xFF) shl 8) or (n * 4)
        }
        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }

    /**
     * A cartoon "portrait": skin-toned oval with eyes and mouth on a grey background, plus seeded
     * noise so smoothing filters have something to remove. Fully deterministic.
     */
    fun syntheticPortrait(width: Int = 512, height: Int = 640, seed: Int = 7): Bitmap {
        val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bmp)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG)
        canvas.drawColor(Color.rgb(96, 104, 112))
        paint.color = Color.rgb(214, 170, 140)
        canvas.drawOval(RectF(width * 0.2f, height * 0.12f, width * 0.8f, height * 0.88f), paint)
        paint.color = Color.rgb(60, 40, 30)
        canvas.drawOval(RectF(width * 0.33f, height * 0.38f, width * 0.43f, height * 0.44f), paint)
        canvas.drawOval(RectF(width * 0.57f, height * 0.38f, width * 0.67f, height * 0.44f), paint)
        paint.color = Color.rgb(170, 80, 80)
        canvas.drawOval(RectF(width * 0.4f, height * 0.66f, width * 0.6f, height * 0.72f), paint)

        val random = Random(seed)
        val pixels = IntArray(width * height).also { bmp.getPixels(it, 0, width, 0, 0, width, height) }
        for (i in pixels.indices) {
            val n = random.nextInt(-12, 13)
            val p = pixels[i]
            fun ch(shift: Int) = (((p shr shift) and 0xFF) + n).coerceIn(0, 255)
            pixels[i] = (0xFF shl 24) or (ch(16) shl 16) or (ch(8) shl 8) or ch(0)
        }
        bmp.setPixels(pixels, 0, width, 0, 0, width, height)
        return bmp
    }

    /** InsightFace ArcFace 5-point template (112 px space) scaled and centred at ([centerX], [centerY]). */
    fun syntheticKeypoints(centerX: Float, centerY: Float, scale: Float): FloatArray {
        val arcface = floatArrayOf(38.29f, 51.70f, 73.53f, 51.50f, 56.03f, 71.74f, 41.55f, 92.37f, 70.73f, 92.20f)
        return FloatArray(10) { i ->
            if (i % 2 == 0) centerX + (arcface[i] - 56f) * scale else centerY + (arcface[i] - 72f) * scale
        }
    }

    /** Seeded, L2-normalised 512-d embedding. */
    fun embedding(seed: Int): FloatArray {
        val random = Random(seed)
        val v = FloatArray(512) { random.nextFloat() - 0.5f }
        val n = sqrt(v.fold(0f) { a, x -> a + x * x })
        return FloatArray(v.size) { v[it] / n }
    }

    /** A [FaceAnalyzer] that "detects" one face at fixed landmarks, so tests need no detector model. */
    class FixedFaceAnalyzer(
        private val keypoints: FloatArray,
        private val box: FaceBoundingBox = FaceBoundingBox(700, 300, 1300, 1100, 0.9f),
        private val age: Int? = null,
        embeddingSeed: Int = 7,
    ) : FaceAnalyzer {
        private val embedding = embedding(embeddingSeed)

        override suspend fun analyze(bitmap: Bitmap, includeEmbedding: Boolean, includeGender: Boolean) =
            FaceAnalysisSummary(
                detectedFaces = 1,
                statusMessage = "fixed",
                embedding = embedding,
                embeddingLength = embedding.size,
                landmarkFivePoints = keypoints,
                allDetectedFaces = listOf(DetectedFaceResult(0, box, keypoints, null, age = age)),
            )
    }
}
