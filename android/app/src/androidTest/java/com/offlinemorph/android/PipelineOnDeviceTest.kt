package com.offlinemorph.android

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.offlinemorph.android.core.ml.OnDeviceFaceSwapEngine
import com.offlinemorph.android.core.ml.OrtSessionFactory
import com.offlinemorph.android.core.ml.SwapRequest
import com.offlinemorph.android.feature.models.ModelCatalog
import java.io.File
import java.nio.FloatBuffer
import java.util.EnumSet
import kotlin.random.Random
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader

/**
 * On-device checks that need the real models installed in the app's models directory
 * (they are skipped otherwise). Results are logged under the `OMBench` tag.
 *
 * Run without uninstalling the app (connectedAndroidTest would wipe the downloaded models):
 *   ./gradlew :app:installDebug :app:installDebugAndroidTest
 *   adb shell am instrument -w -e class com.offlinemorph.android.PipelineOnDeviceTest \
 *     com.offlinemorph.android.test/androidx.test.runner.AndroidJUnitRunner
 */
@RunWith(AndroidJUnit4::class)
class PipelineOnDeviceTest {

    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val modelsDir = File(context.filesDir, "models")

    @Before
    fun setUp() {
        assumeTrue("Models not installed", File(modelsDir, ModelCatalog.INSWAPPER).isFile)
        OpenCVLoader.initLocal()
    }

    /**
     * Runs the real swap pipeline (inswapper, pixel boost, paste-back, GFPGAN) on a synthetic
     * image with injected landmarks, so it needs no photos of real people.
     */
    @Test
    fun swapPipeline_runsEndToEnd() = runBlocking {
        val target = TestFixtures.noiseBitmap(2048, 1536)
        val keypoints = TestFixtures.syntheticKeypoints(centerX = 1000f, centerY = 700f, scale = 6f)
        val engine = OnDeviceFaceSwapEngine(modelsDir, TestFixtures.FixedFaceAnalyzer(keypoints), OrtSessionFactory())

        val configs = listOf(
            "boost1" to SwapRequest(target, target, seamlessBlend = false),
            "boost1+poisson" to SwapRequest(target, target),
            "boost2+poisson" to SwapRequest(target, target, pixelBoost = 2),
            "boost2+poisson+gfpgan" to SwapRequest(target, target, pixelBoost = 2, enhancerEnabled = true),
        )
        for ((name, request) in configs) {
            repeat(2) { attempt ->
                val start = System.nanoTime()
                val result = engine.runSwap(request)
                val ms = (System.nanoTime() - start) / 1_000_000
                Log.i(TAG, "pipeline $name run${attempt + 1}: $ms ms — ${result.statusMessage}")
                val output = assertNotNull(result.statusMessage, result.outputBitmap).let { result.outputBitmap!! }
                assertEquals(target.width, output.width)
                assertEquals(target.height, output.height)
                // Pixels far from the face must be untouched; pixels at the face must change.
                assertEquals(target.getPixel(10, 10), output.getPixel(10, 10))
                assert(target.getPixel(1000, 700) != output.getPixel(1000, 700)) { "Face region unchanged" }
            }
        }
    }

    /** Compares execution providers per model: session load time and median inference time. */
    @Test
    fun executionProviderBenchmark() {
        val env = OrtEnvironment.getEnvironment()
        val models = listOf(
            ModelCatalog.DETECTOR to listOf(longArrayOf(1, 3, 640, 640)),
            ModelCatalog.RECOGNIZER to listOf(longArrayOf(1, 3, 112, 112)),
            ModelCatalog.INSWAPPER to listOf(longArrayOf(1, 3, 128, 128), longArrayOf(1, 512)),
            ModelCatalog.GFPGAN to listOf(longArrayOf(1, 3, 512, 512)),
        )
        val providers: List<Pair<String, OrtSession.SessionOptions.() -> Unit>> = listOf(
            "cpu" to {},
            "nnapi" to { addNnapi() },
            "nnapi-fp16-nocpu" to { addNnapi(EnumSet.of(NNAPIFlags.USE_FP16, NNAPIFlags.CPU_DISABLED)) },
            "cpu-4threads" to { setIntraOpNumThreads(4) },
            // Not benchmarked: XNNPACK aborts natively on det_10g; QNN is not in this ORT build
            // (see OrtSessionFactory for the NPU findings).
        )
        for ((fileName, shapes) in models) {
            val file = File(modelsDir, fileName)
            if (!file.isFile) continue
            for ((provider, configure) in providers) {
                runCatching {
                    val loadStart = System.nanoTime()
                    val options = OrtSession.SessionOptions().apply {
                        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                        configure()
                    }
                    env.createSession(file.absolutePath, options).use { session ->
                        val loadMs = (System.nanoTime() - loadStart) / 1_000_000
                        val tensors = session.inputNames.zip(shapes).associate { (name, shape) ->
                            name to OnnxTensor.createTensor(env, FloatBuffer.wrap(randomData(shape)), shape)
                        }
                        session.run(tensors).close() // warm-up
                        val runs = List(5) {
                            val t = System.nanoTime()
                            session.run(tensors).close()
                            (System.nanoTime() - t) / 1_000_000
                        }.sorted()
                        tensors.values.forEach { it.close() }
                        Log.i(TAG, "bench %-20s %-17s load=%6d ms  median=%5d ms  min=%5d ms".format(
                            fileName, provider, loadMs, runs[2], runs[0]))
                    }
                }.onFailure { Log.i(TAG, "bench %-20s %-17s FAILED: %s".format(fileName, provider, it.message)) }
            }
        }
    }

    private fun randomData(shape: LongArray): FloatArray {
        val random = Random(3)
        return FloatArray(shape.fold(1L) { a, d -> a * d }.toInt()) { random.nextFloat() }
    }

    private companion object {
        const val TAG = "OMBench"
    }
}
