package com.offlinemorph.android

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.io.FileNotFoundException
import kotlin.math.abs
import kotlin.math.max
import org.junit.Assert.assertEquals
import org.junit.Assert.fail
import org.junit.Assume.assumeTrue

/**
 * Golden-image comparison for instrumented tests.
 *
 * Reference images live in the test APK at `app/src/androidTest/assets/goldens/<name>.png`.
 *
 * **Recording.** When a golden is missing, or the run passes `-e recordGoldens true`, the actual
 * output is written to the app's external files directory and the test is skipped (not failed):
 *
 *   adb shell am instrument -w -e recordGoldens true \
 *     -e class com.offlinemorph.android.GoldenImageRegressionTest \
 *     com.offlinemorph.android.test/androidx.test.runner.AndroidJUnitRunner
 *   adb pull /sdcard/Android/data/com.offlinemorph.android/files/goldens/. \
 *     app/src/androidTest/assets/goldens/
 *
 * **Comparison.** ONNX Runtime and OpenCV results differ slightly between CPUs, so pixels are
 * compared with a tolerance: the mean absolute channel difference must stay under
 * [Tolerance.maxMeanAbsDiff] and at most [Tolerance.maxOutlierFraction] of pixels may differ by
 * more than [Tolerance.outlierThreshold] in any channel. On failure the actual image and an
 * amplified diff are written to `files/golden-failures/` for inspection.
 */
object GoldenImage {
    private const val TAG = "GoldenImage"
    private const val ASSET_DIR = "goldens"

    data class Tolerance(
        val maxMeanAbsDiff: Double = 1.0,
        val outlierThreshold: Int = 24,
        val maxOutlierFraction: Double = 0.002,
    )

    /** Pixel-exact-ish: for pure CPU image ops with no ML inference. */
    val STRICT = Tolerance(maxMeanAbsDiff = 0.5, outlierThreshold = 8, maxOutlierFraction = 0.001)

    /** For ML outputs, which vary a little across devices' float kernels. */
    val ML = Tolerance(maxMeanAbsDiff = 1.5, outlierThreshold = 32, maxOutlierFraction = 0.005)

    private val instrumentation get() = InstrumentationRegistry.getInstrumentation()

    private fun outputDir(sub: String): File =
        File(requireNotNull(instrumentation.targetContext.getExternalFilesDir(null)), sub).apply { mkdirs() }

    private val recording: Boolean
        get() = InstrumentationRegistry.getArguments().getString("recordGoldens") == "true"

    fun assertMatches(name: String, actual: Bitmap, tolerance: Tolerance = STRICT) {
        val golden = loadGolden(name)
        if (golden == null || recording) {
            val file = File(outputDir(ASSET_DIR), "$name.png")
            save(actual, file)
            Log.i(TAG, "Recorded golden $name → ${file.absolutePath}")
            assumeTrue(
                "Golden '$name' ${if (golden == null) "missing" else "re-recorded"}; wrote ${file.absolutePath}. " +
                    "Copy it to app/src/androidTest/assets/$ASSET_DIR/ and re-run.",
                false,
            )
            return
        }

        assertEquals("$name: width", golden.width, actual.width)
        assertEquals("$name: height", golden.height, actual.height)
        val stats = compare(golden, actual, tolerance.outlierThreshold)
        Log.i(TAG, "$name: $stats")
        if (stats.meanAbsDiff > tolerance.maxMeanAbsDiff || stats.outlierFraction > tolerance.maxOutlierFraction) {
            val dir = outputDir("golden-failures")
            save(actual, File(dir, "$name.actual.png"))
            save(stats.diffImage, File(dir, "$name.diff.png"))
            fail("$name differs from golden: $stats (limits: mean ≤ ${tolerance.maxMeanAbsDiff}, " +
                "outliers ≤ ${tolerance.maxOutlierFraction}). Actual and diff written to ${dir.absolutePath}")
        }
    }

    data class DiffStats(
        val meanAbsDiff: Double,
        val maxChannelDiff: Int,
        val outlierFraction: Double,
        val diffImage: Bitmap,
    ) {
        override fun toString() =
            "meanAbsDiff=%.3f maxChannelDiff=%d outlierFraction=%.5f".format(meanAbsDiff, maxChannelDiff, outlierFraction)
    }

    fun compare(expected: Bitmap, actual: Bitmap, outlierThreshold: Int): DiffStats {
        val w = expected.width
        val h = expected.height
        val e = IntArray(w * h).also { expected.getPixels(it, 0, w, 0, 0, w, h) }
        val a = IntArray(w * h).also { actual.getPixels(it, 0, w, 0, 0, w, h) }
        val diff = IntArray(w * h)
        var sum = 0L
        var maxDiff = 0
        var outliers = 0
        for (i in e.indices) {
            val dr = abs(((e[i] shr 16) and 0xFF) - ((a[i] shr 16) and 0xFF))
            val dg = abs(((e[i] shr 8) and 0xFF) - ((a[i] shr 8) and 0xFF))
            val db = abs((e[i] and 0xFF) - (a[i] and 0xFF))
            val m = max(dr, max(dg, db))
            sum += dr + dg + db
            if (m > maxDiff) maxDiff = m
            if (m > outlierThreshold) outliers++
            val v = (m * 8).coerceAtMost(255)
            diff[i] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
        }
        return DiffStats(
            meanAbsDiff = sum.toDouble() / (e.size * 3),
            maxChannelDiff = maxDiff,
            outlierFraction = outliers.toDouble() / e.size,
            diffImage = Bitmap.createBitmap(diff, w, h, Bitmap.Config.ARGB_8888),
        )
    }

    private fun loadGolden(name: String): Bitmap? = try {
        instrumentation.context.assets.open("$ASSET_DIR/$name.png").use { stream ->
            BitmapFactory.decodeStream(stream, null, BitmapFactory.Options().apply {
                inPreferredConfig = Bitmap.Config.ARGB_8888
            })
        }
    } catch (_: FileNotFoundException) {
        null
    }

    private fun save(bitmap: Bitmap, file: File) {
        file.outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
    }
}
