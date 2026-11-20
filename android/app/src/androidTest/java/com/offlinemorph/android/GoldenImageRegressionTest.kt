package com.offlinemorph.android

import android.graphics.Bitmap
import android.graphics.Rect
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.offlinemorph.android.core.ml.EngineError
import com.offlinemorph.android.core.ml.EngineResult
import com.offlinemorph.android.core.ml.FaceAlignmentOps
import com.offlinemorph.android.core.ml.FaceBoundingBox
import com.offlinemorph.android.core.ml.OnDeviceFaceSwapEngine
import com.offlinemorph.android.core.ml.OrtSessionFactory
import com.offlinemorph.android.core.ml.SwapRequest
import com.offlinemorph.android.core.ml.aging.AgingRequest
import com.offlinemorph.android.core.ml.aging.OnDeviceAgingEngine
import com.offlinemorph.android.core.ml.ancestry.OnDeviceAncestryEngine
import com.offlinemorph.android.core.ml.beautify.BeautifyRequest
import com.offlinemorph.android.core.ml.beautify.OnDeviceBeautifyEngine
import com.offlinemorph.android.feature.models.ModelCatalog
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar

/**
 * Golden-image regression suite for image outputs (Sprint 1, item 5).
 *
 * - CPU-only pipelines (beautify, paste-back blending) always run.
 * - Swap and aging run only when their models are installed in the app's `files/models`
 *   directory, and are skipped otherwise. Run them on a device that has the models, without
 *   uninstalling the app (connectedAndroidTest would wipe the downloaded models):
 *
 *     ./gradlew :app:installDebug :app:installDebugAndroidTest
 *     adb shell am instrument -w -e class com.offlinemorph.android.GoldenImageRegressionTest \
 *       com.offlinemorph.android.test/androidx.test.runner.AndroidJUnitRunner
 *
 * See [GoldenImage] for recording new or updated goldens.
 */
@RunWith(AndroidJUnit4::class)
class GoldenImageRegressionTest {

    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val modelsDir = File(context.filesDir, "models")

    @Before
    fun setUp() {
        OpenCVLoader.initLocal()
    }

    // ── CPU-only pipelines ───────────────────────────────────────────────────

    @Test
    fun beautify_skinSmoothing_matchesGolden() = runBlocking {
        val result = OnDeviceBeautifyEngine().runBeautify(
            BeautifyRequest(sourceBitmap = TestFixtures.syntheticPortrait(), skinSmoothing = 0.7f),
        )
        val output = (result as EngineResult.Success).value.outputBitmap
        GoldenImage.assertMatches("beautify_skin_smoothing_070", output, GoldenImage.STRICT)
    }

    @Test
    fun beautify_zeroSmoothing_isIdentity() = runBlocking {
        val source = TestFixtures.syntheticPortrait()
        val result = OnDeviceBeautifyEngine().runBeautify(BeautifyRequest(sourceBitmap = source))
        val output = (result as EngineResult.Success).value.outputBitmap
        val stats = GoldenImage.compare(source, output, outlierThreshold = 0)
        assertEquals("Zero smoothing must not change any pixel: $stats", 0, stats.maxChannelDiff)
    }

    @Test
    fun pasteBack_feathered_matchesGolden() {
        val (target, patch, toTarget) = pasteFixture()
        val output = FaceAlignmentOps.pasteBack(target, patch, toTarget, seamless = false)
        patch.release()
        GoldenImage.assertMatches("paste_back_feathered", output, GoldenImage.STRICT)
    }

    @Test
    fun pasteBack_poisson_matchesGolden() {
        val (target, patch, toTarget) = pasteFixture()
        val output = FaceAlignmentOps.pasteBack(target, patch, toTarget, seamless = true)
        patch.release()
        GoldenImage.assertMatches("paste_back_poisson", output, GoldenImage.STRICT)
    }

    // ── Ancestry (no model needed for the identity math) ─────────────────────

    @Test
    fun ancestry_blendEmbeddings_interpolatesOnTheSphere() {
        val a = TestFixtures.embedding(1)
        val b = TestFixtures.embedding(2)
        assertTrue(cosine(OnDeviceAncestryEngine.blendEmbeddings(a, b, 1f), a) > 0.9999f)
        assertTrue(cosine(OnDeviceAncestryEngine.blendEmbeddings(a, b, 0f), b) > 0.9999f)
        val mid = OnDeviceAncestryEngine.blendEmbeddings(a, b, 0.5f)
        assertEquals(1f, norm(mid), 1e-4f)
        assertEquals("Midpoint must be equidistant", cosine(mid, a), cosine(mid, b), 1e-4f)
        val nearA = OnDeviceAncestryEngine.blendEmbeddings(a, b, 0.8f)
        assertTrue(cosine(nearA, a) > cosine(nearA, b))
    }

    @Test
    fun ancestry_missingModels_reportsModelNotFound() = runBlocking {
        val emptyDir = File(context.cacheDir, "no-models").apply { mkdirs() }
        val analyzer = TestFixtures.FixedFaceAnalyzer(TestFixtures.syntheticKeypoints(256f, 256f, 2f))
        val engine = OnDeviceAncestryEngine(emptyDir, analyzer, OnDeviceFaceSwapEngine(emptyDir, analyzer, OrtSessionFactory()))
        val portrait = TestFixtures.syntheticPortrait()
        val result = engine.runAncestry(
            com.offlinemorph.android.core.ml.ancestry.AncestryRequest(portrait, portrait, portrait),
        )
        assertTrue((result as EngineResult.Failure).error is EngineError.ModelNotFound)
    }

    @Test
    fun aging_missingModel_reportsModelNotFound() = runBlocking {
        val emptyDir = File(context.cacheDir, "no-models").apply { mkdirs() }
        val analyzer = TestFixtures.FixedFaceAnalyzer(TestFixtures.syntheticKeypoints(256f, 256f, 2f))
        val result = OnDeviceAgingEngine(emptyDir, analyzer).runAging(
            AgingRequest(TestFixtures.syntheticPortrait(), ageOffsetYears = 20),
        )
        assertTrue((result as EngineResult.Failure).error is EngineError.ModelNotFound)
    }

    // ── Model-backed pipelines (skipped when models are not installed) ───────

    @Test
    fun swap_inswapperPoisson_matchesGolden() = runBlocking {
        assumeTrue("inswapper not installed", File(modelsDir, ModelCatalog.INSWAPPER).isFile)
        val target = TestFixtures.noiseBitmap(1024, 768)
        val analyzer = TestFixtures.FixedFaceAnalyzer(
            keypoints = TestFixtures.syntheticKeypoints(centerX = 512f, centerY = 384f, scale = 3f),
            box = SWAP_FACE.toBox(),
        )
        val engine = OnDeviceFaceSwapEngine(modelsDir, analyzer, OrtSessionFactory())
        val output = engine.runSwap(SwapRequest(sourceBitmap = target, targetBitmap = target)).outputBitmap
        GoldenImage.assertMatches("swap_inswapper_poisson", faceCrop(output!!, SWAP_FACE), GoldenImage.ML)
    }

    @Test
    fun aging_franPlus30_matchesGolden() = runBlocking {
        assumeTrue("fran not installed", File(modelsDir, ModelCatalog.FRAN).isFile)
        val target = TestFixtures.noiseBitmap(1024, 1024, seed = 5)
        val analyzer = TestFixtures.FixedFaceAnalyzer(
            keypoints = TestFixtures.syntheticKeypoints(centerX = 512f, centerY = 512f, scale = 4f),
            box = AGING_FACE.toBox(),
            age = 30,
        )
        val result = OnDeviceAgingEngine(modelsDir, analyzer).runAging(
            AgingRequest(sourceBitmap = target, ageOffsetYears = 30),
        )
        val output = (result as EngineResult.Success).value
        assertEquals(30, output.appliedAgeOffset)
        GoldenImage.assertMatches("aging_fran_plus30", faceCrop(output.outputBitmap, AGING_FACE), GoldenImage.ML)
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    /** A 256 px patch (vertical colour bars) placed rotated and scaled into a noisy 640×480 target. */
    private fun pasteFixture(): Triple<Bitmap, Mat, android.graphics.Matrix> {
        val target = TestFixtures.noiseBitmap(640, 480, seed = 3)
        val size = 256
        val patch = Mat(size, size, CvType.CV_8UC3, Scalar(200.0, 120.0, 60.0))
        val bytes = ByteArray(size * size * 3).also { patch.get(0, 0, it) }
        for (y in 0 until size) for (x in 0 until size) {
            if ((x / 32) % 2 == 0) {
                val i = (y * size + x) * 3
                bytes[i] = 40; bytes[i + 1] = (x % 256).toByte(); bytes[i + 2] = (y % 256).toByte()
            }
        }
        patch.put(0, 0, bytes)
        val toTarget = android.graphics.Matrix().apply {
            setRotate(12f, size / 2f, size / 2f)
            postScale(1.1f, 1.1f)
            postTranslate(180f, 90f)
        }
        return Triple(target, patch, toTarget)
    }

    /** Crops [rect] and scales it to 256 px so goldens stay small and focus on the face. */
    private fun faceCrop(bitmap: Bitmap, rect: Rect): Bitmap {
        val crop = Bitmap.createBitmap(bitmap, rect.left, rect.top, rect.width(), rect.height())
        return Bitmap.createScaledBitmap(crop, 256, 256 * rect.height() / rect.width(), true)
    }

    private fun Rect.toBox() = FaceBoundingBox(left, top, right, bottom, 0.9f)

    private fun cosine(a: FloatArray, b: FloatArray): Float =
        a.indices.fold(0f) { acc, i -> acc + a[i] * b[i] } / (norm(a) * norm(b))

    private fun norm(v: FloatArray): Float = kotlin.math.sqrt(v.fold(0f) { acc, x -> acc + x * x })

    private companion object {
        val SWAP_FACE = Rect(320, 144, 704, 624)
        val AGING_FACE = Rect(256, 192, 768, 832)
    }
}
