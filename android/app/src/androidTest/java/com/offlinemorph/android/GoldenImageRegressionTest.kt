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
import com.offlinemorph.android.core.ml.beautify.BeautifyRequest
import com.offlinemorph.android.core.ml.beautify.OnDeviceBeautifyEngine
import com.offlinemorph.android.core.ml.face.ColorLab
import com.offlinemorph.android.core.ml.face.FaceWarp
import com.offlinemorph.android.core.ml.face.RegionEffects
import com.offlinemorph.android.core.ml.hairmakeup.FacialHairStyle
import com.offlinemorph.android.core.ml.hairmakeup.HairMakeupRequest
import com.offlinemorph.android.core.ml.hairmakeup.HairStyle
import com.offlinemorph.android.core.ml.hairmakeup.OnDeviceHairMakeupEngine
import com.offlinemorph.android.core.video.KeypointStabilizer
import com.offlinemorph.android.feature.device.QualityProfile
import com.offlinemorph.android.feature.swap.effectiveProfile
import com.offlinemorph.android.feature.swap.swapSettings
import kotlin.math.abs
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
 * - CPU-only pipelines (beautify with the landmark-template parse, paste-back blending, colour
 *   and warp primitives) always run.
 * - Swap, aging, and hair & makeup run only when their models are installed in the app's `files/models`
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

    /** Beautify with the landmark-template parse (no parsing model) on the cartoon portrait. */
    private fun beautifyEngine() = OnDeviceBeautifyEngine(
        modelsDirectory = detectorOnlyDir(),
        faceAnalyzer = TestFixtures.FixedFaceAnalyzer(PORTRAIT_KEYPOINTS, box = PORTRAIT_BOX),
    )

    @Test
    fun beautify_skinSmoothing_matchesGolden() = runBlocking {
        val result = beautifyEngine().runBeautify(
            BeautifyRequest(sourceBitmap = TestFixtures.syntheticPortrait(), skinSmoothing = 0.7f),
        )
        val output = (result as EngineResult.Success).value.outputBitmap
        GoldenImage.assertMatches("beautify_skin_070", output, GoldenImage.STRICT)
    }

    @Test
    fun beautify_allControls_matchesGolden() = runBlocking {
        val result = beautifyEngine().runBeautify(
            BeautifyRequest(
                sourceBitmap = TestFixtures.syntheticPortrait(),
                skinSmoothing = 0.6f, eyeEnlarge = 0.8f, faceSlim = 0.8f, teethWhiten = 1f,
            ),
        )
        val output = (result as EngineResult.Success).value.outputBitmap
        GoldenImage.assertMatches("beautify_all_controls", output, GoldenImage.STRICT)
    }

    @Test
    fun beautify_skinSmoothing_leavesBackgroundUntouched() = runBlocking {
        val source = TestFixtures.syntheticPortrait()
        val output = (beautifyEngine().runBeautify(BeautifyRequest(source, skinSmoothing = 1f))
            as EngineResult.Success).value.outputBitmap
        assertEquals("Corner (background) must not change", source.getPixel(8, 8), output.getPixel(8, 8))
        val cheek = 256 to 400
        assertTrue("Skin must be smoothed", source.getPixel(cheek.first, cheek.second) != output.getPixel(cheek.first, cheek.second))
    }

    @Test
    fun beautify_faceEditsWithoutDetector_reportModelNotFound() = runBlocking {
        val result = OnDeviceBeautifyEngine(
            File(context.cacheDir, "no-models").apply { mkdirs() },
            TestFixtures.FixedFaceAnalyzer(PORTRAIT_KEYPOINTS),
        ).runBeautify(BeautifyRequest(TestFixtures.syntheticPortrait(), eyeEnlarge = 0.5f))
        assertTrue((result as EngineResult.Failure).error is EngineError.ModelNotFound)
    }

    @Test
    fun beautify_zeroSmoothing_isIdentity() = runBlocking {
        val source = TestFixtures.syntheticPortrait()
        val result = beautifyEngine().runBeautify(BeautifyRequest(sourceBitmap = source))
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

    // ── Colour and warp primitives ───────────────────────────────────────────

    @Test
    fun colorLab_roundTrips() {
        val lab = FloatArray(3)
        for (rgb in listOf(0x000000, 0xFFFFFF, 0xC0392B, 0x1A3A8B, 0xE8E0D0, 0x6B3A2A)) {
            ColorLab.toLab(rgb, lab)
            val back = ColorLab.toRgb(lab[0], lab[1], lab[2])
            for (shift in listOf(16, 8, 0)) {
                assertTrue("0x%06x → 0x%06x".format(rgb, back), abs(((rgb shr shift) and 0xFF) - ((back shr shift) and 0xFF)) <= 1)
            }
        }
    }

    @Test
    fun faceWarp_withoutOperations_isIdentity() {
        val rgb = Mat(40, 60, CvType.CV_8UC3)
        org.opencv.core.Core.randu(rgb, 0.0, 255.0)
        val warp = FaceWarp(60, 40)
        assertTrue(warp.isEmpty)
        val out = warp.apply(rgb)
        val diff = Mat()
        org.opencv.core.Core.absdiff(rgb, out, diff)
        assertEquals(0.0, org.opencv.core.Core.sumElems(diff).`val`.sum(), 0.0)
    }

    @Test
    fun whitenTeeth_onlyTouchesToothLikePixels() {
        // Left half: yellowish tooth colour; right half: saturated red lip colour.
        val rgb = Mat(10, 20, CvType.CV_8UC3)
        val bytes = ByteArray(10 * 20 * 3)
        for (i in 0 until 200) {
            val tooth = (i % 20) < 10
            bytes[i * 3] = (if (tooth) 230 else 180).toByte()
            bytes[i * 3 + 1] = (if (tooth) 215 else 40).toByte()
            bytes[i * 3 + 2] = (if (tooth) 160 else 60).toByte()
        }
        rgb.put(0, 0, bytes)
        val mask = Mat(10, 20, CvType.CV_32FC1, Scalar(1.0))
        RegionEffects.whitenTeeth(rgb, mask, 1f)
        val out = ByteArray(bytes.size).also { rgb.get(0, 0, it) }
        assertTrue("Tooth blue channel must rise (less yellow)", (out[2].toInt() and 0xFF) > 160)
        assertEquals("Lip pixel must be unchanged", bytes[13 * 3].toInt(), out[13 * 3].toInt())
    }

    // ── Hair & makeup (needs face_parsing.onnx) ──────────────────────────────

    private fun hairMakeupEngine(): OnDeviceHairMakeupEngine {
        assumeTrue("face_parsing not installed", File(modelsDir, ModelCatalog.FACE_PARSING).isFile)
        assumeTrue("detector not installed", File(modelsDir, ModelCatalog.DETECTOR).isFile)
        return OnDeviceHairMakeupEngine(modelsDir, TestFixtures.FixedFaceAnalyzer(PORTRAIT_KEYPOINTS, box = PORTRAIT_BOX))
    }

    @Test
    fun makeup_fullLook_matchesGolden() = runBlocking {
        val result = hairMakeupEngine().runHairMakeup(
            HairMakeupRequest(
                sourceBitmap = TestFixtures.syntheticPortrait(),
                lipColorHex = "c62828", eyeshadowColorHex = "6e3a5e", eyelinerColorHex = "141414",
                blushColorHex = "f08da0", facialHair = FacialHairStyle.FULL_BEARD,
            ),
        )
        val output = (result as EngineResult.Success).value.outputBitmap
        GoldenImage.assertMatches("makeup_full_look", output, GoldenImage.ML)
    }

    @Test
    fun hair_ombre_matchesGolden() = runBlocking {
        val result = hairMakeupEngine().runHairMakeup(
            HairMakeupRequest(
                sourceBitmap = TestFixtures.syntheticPortrait(),
                hairColorHex = "3b2314", hairStyle = HairStyle.OMBRE, hairSecondaryColorHex = "e3cc95",
            ),
        )
        val output = (result as EngineResult.Success).value.outputBitmap
        GoldenImage.assertMatches("hair_ombre", output, GoldenImage.ML)
    }

    // ── Swap quality and video stabilisation ─────────────────────────────────

    @Test
    fun qualityProfile_policyCanOnlyLower() {
        assertEquals(QualityProfile.FAST, effectiveProfile(QualityProfile.STUDIO, QualityProfile.FAST))
        assertEquals(QualityProfile.BALANCED, effectiveProfile(QualityProfile.BALANCED, QualityProfile.STUDIO))
        assertEquals(2, QualityProfile.STUDIO.swapSettings().pixelBoost)
        assertTrue(!QualityProfile.FAST.swapSettings().occlusionAware)
    }

    @Test
    fun keypointStabilizer_dampsJitterButFollowsMotion() {
        val stabilizer = KeypointStabilizer()
        val base = TestFixtures.syntheticKeypoints(500f, 400f, 3f)
        val random = kotlin.random.Random(9)
        // Still face with ±2 px detector jitter: output jitter must shrink.
        var rawJitter = 0f; var outJitter = 0f
        var previous = stabilizer.stabilize(listOf(base)).first()
        repeat(30) {
            val noisy = FloatArray(10) { i -> base[i] + random.nextFloat() * 4f - 2f }
            val out = stabilizer.stabilize(listOf(noisy)).first()
            rawJitter += abs(noisy[0] - base[0]); outJitter += abs(out[0] - previous[0])
            previous = out
        }
        assertTrue("jitter raw=$rawJitter smoothed=$outJitter", outJitter < rawJitter * 0.6f)
        // A fast move (40 px, ~0.4 eye distances) is followed within one frame.
        val moved = FloatArray(10) { i -> base[i] + if (i % 2 == 0) 40f else 0f }
        val out = stabilizer.stabilize(listOf(moved)).first()
        assertEquals(moved[0], out[0], 1.5f)
    }

    // ── Missing-model contracts ──────────────────────────────────────────────

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
    fun swap_occlusionAware_matchesGolden() = runBlocking {
        assumeTrue("inswapper not installed", File(modelsDir, ModelCatalog.INSWAPPER).isFile)
        assumeTrue("face_parsing not installed", File(modelsDir, ModelCatalog.FACE_PARSING).isFile)
        val target = TestFixtures.noiseBitmap(1024, 768)
        val analyzer = TestFixtures.FixedFaceAnalyzer(
            keypoints = TestFixtures.syntheticKeypoints(centerX = 512f, centerY = 384f, scale = 3f),
            box = SWAP_FACE.toBox(),
        )
        val engine = OnDeviceFaceSwapEngine(modelsDir, analyzer, OrtSessionFactory())
        val output = engine.runSwap(SwapRequest(target, target, occlusionAware = true)).outputBitmap
        GoldenImage.assertMatches("swap_occlusion_aware", faceCrop(output!!, SWAP_FACE), GoldenImage.ML)
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

    /** A models directory with only a placeholder detector: engines see "detector installed",
     *  while the fixed analyzer means the placeholder is never loaded. */
    private fun detectorOnlyDir(): File = File(context.cacheDir, "detector-only").apply {
        mkdirs()
        File(this, ModelCatalog.DETECTOR).writeText("placeholder")
    }

    private companion object {
        val SWAP_FACE = Rect(320, 144, 704, 624)
        val AGING_FACE = Rect(256, 192, 768, 832)

        /** Eyes, nose and mouth corners of [TestFixtures.syntheticPortrait] (512 × 640). */
        val PORTRAIT_KEYPOINTS = floatArrayOf(195f, 262f, 317f, 262f, 256f, 350f, 215f, 442f, 297f, 442f)
        val PORTRAIT_BOX = FaceBoundingBox(102, 77, 410, 563, 0.9f)
    }
}
