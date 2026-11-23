package com.offlinemorph.android

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.offlinemorph.android.core.ml.OrtValueUtils
import com.offlinemorph.android.feature.models.ModelCatalog
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt
import kotlin.random.Random
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Diagnostic: runs every installed model on the CPU and on the Hexagon NPU (QNN HTP, FP16) with
 * identical inputs and logs, per model, session load time, median latency and how closely the
 * NPU output matches the CPU output. Results are logged under the `OMNpu` tag; nothing is asserted.
 *
 *   adb shell am instrument -w -e class com.offlinemorph.android.NpuBenchmarkTest \
 *     com.offlinemorph.android.test/androidx.test.runner.AndroidJUnitRunner
 *
 * Optional: `-e npuModels det_10g.onnx,fran.onnx` limits the run to those files.
 */
@RunWith(AndroidJUnit4::class)
class NpuBenchmarkTest {

    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val modelsDir = File(context.filesDir, "models")
    private val env = OrtEnvironment.getEnvironment()

    /** Input name → (shape, value generator) per model; dynamic dims are fixed here. */
    private data class Spec(val file: String, val inputs: (OrtSession) -> Map<String, Pair<LongArray, FloatArray>>)

    private fun image(shape: LongArray, lo: Float, hi: Float, seed: Int): FloatArray {
        val r = Random(seed)
        return FloatArray(shape.fold(1L) { a, d -> a * d }.toInt()) { lo + (hi - lo) * r.nextFloat() }
    }

    private fun single(shape: LongArray, lo: Float, hi: Float): (OrtSession) -> Map<String, Pair<LongArray, FloatArray>> =
        { s -> mapOf(s.inputNames.first() to (shape to image(shape, lo, hi, 1))) }

    private val specs = listOf(
        Spec(ModelCatalog.DETECTOR, single(longArrayOf(1, 3, 640, 640), -1f, 1f)),
        Spec(ModelCatalog.RECOGNIZER, single(longArrayOf(1, 3, 112, 112), -1f, 1f)),
        Spec(ModelCatalog.GENDERAGE, single(longArrayOf(1, 3, 96, 96), 0f, 255f)),
        Spec(ModelCatalog.LANDMARKS_2D, single(longArrayOf(1, 3, 192, 192), 0f, 255f)),
        Spec(ModelCatalog.FACE_PARSING, single(longArrayOf(1, 3, 512, 512), -2f, 2f)),
        Spec(ModelCatalog.GFPGAN, single(longArrayOf(1, 3, 512, 512), -1f, 1f)),
        Spec(ModelCatalog.INSWAPPER) { s ->
            val names = s.inputNames.toList()
            val target = names.firstOrNull { it == "target" } ?: names[0]
            val source = names.firstOrNull { it == "source" } ?: names[1]
            val latent = image(longArrayOf(1, 512), -1f, 1f, 2).let { v ->
                val n = sqrt(v.fold(0f) { a, x -> a + x * x }); FloatArray(v.size) { v[it] / n }
            }
            mapOf(
                target to (longArrayOf(1, 3, 128, 128) to image(longArrayOf(1, 3, 128, 128), 0f, 1f, 1)),
                source to (longArrayOf(1, 512) to latent),
            )
        },
        Spec(ModelCatalog.FRAN) { s ->
            s.inputNames.associateWith { name ->
                if (name.contains("direction")) {
                    val rank = (s.inputInfo[name]?.info as? TensorInfo)?.shape?.size ?: 1
                    (if (rank == 2) longArrayOf(1, 2) else longArrayOf(2)) to floatArrayOf(0.3f, 0.6f)
                } else {
                    longArrayOf(1, 3, 1024, 1024) to image(longArrayOf(1, 3, 1024, 1024), 0f, 1f, 1)
                }
            }
        },
    )

    @Test
    fun cpuVsNpu() {
        val filter = InstrumentationRegistry.getArguments().getString("npuModels")?.split(',')?.map { it.trim() }
        val selected = specs.filter { File(modelsDir, it.file).isFile && (filter == null || it.file in filter) }
        assumeTrue("No models installed", selected.isNotEmpty())
        for (spec in selected) {
            val path = File(modelsDir, spec.file).absolutePath
            val cpu = runCatching { measure(path, spec, npu = false) }
                .onFailure { Log.e(TAG, "${spec.file} CPU failed", it) }.getOrNull() ?: continue
            val npu = runCatching { measure(path, spec, npu = true) }
                .onFailure { Log.e(TAG, "${spec.file} NPU failed: ${it.message}") }.getOrNull()
            if (npu == null) {
                Log.i(TAG, "%-20s cpu load=%5d ms run=%6d ms | npu FAILED".format(spec.file, cpu.loadMs, cpu.medianMs))
                continue
            }
            val acc = compare(cpu.outputs, npu.outputs)
            Log.i(TAG, "%-20s cpu load=%5d ms run=%6d ms | npu load=%6d ms run=%6d ms | %s".format(
                spec.file, cpu.loadMs, cpu.medianMs, npu.loadMs, npu.medianMs, acc))
        }
    }

    /**
     * Compiles a model for the NPU once with ONNX Runtime's EP-context cache, then reloads the
     * cached compiled graph and checks it loads fast and gives the same output.
     */
    @Test
    fun npuContextCache() {
        val file = InstrumentationRegistry.getArguments().getString("cacheModel") ?: ModelCatalog.FACE_PARSING
        val spec = specs.first { it.file == file }
        val model = File(modelsDir, file)
        assumeTrue("$file not installed", model.isFile)
        val ctx = File(context.cacheDir, "npu-ctx-test/${model.nameWithoutExtension}_ctx.onnx").apply {
            parentFile?.mkdirs(); delete()
        }
        val compile = measure(model.absolutePath, spec, npu = true, writeContext = ctx)
        Log.i(TAG, "%-20s compile+write ctx: load=%d ms run=%d ms ctx=%d MB".format(
            file, compile.loadMs, compile.medianMs, ctx.length() / 1_000_000))
        val cached = measure(ctx.absolutePath, spec, npu = true)
        Log.i(TAG, "%-20s from ctx:          load=%d ms run=%d ms | %s".format(
            file, cached.loadMs, cached.medianMs, compare(compile.outputs, cached.outputs)))
    }

    private class Run(val loadMs: Long, val medianMs: Long, val outputs: List<FloatArray>)

    private fun measure(path: String, spec: Spec, npu: Boolean, writeContext: File? = null): Run {
        val options = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            // Pin the models' symbolic dimensions: QNN only accepts static shapes, otherwise the
            // whole graph silently stays on the CPU.
            setSymbolicDimensionValue("batch", 1L)
            setSymbolicDimensionValue("None", 1L)
            setSymbolicDimensionValue("?", 640L)
            if (npu) {
                addQnn(
                    mapOf(
                        "backend_path" to "libQnnHtp.so",
                        "htp_performance_mode" to "burst",
                        "enable_htp_fp16_precision" to "1",
                        "htp_graph_finalization_optimization_mode" to "3",
                    ),
                )
                if (writeContext != null) {
                    addConfigEntry("ep.context_enable", "1")
                    addConfigEntry("ep.context_embed_mode", "1")
                    addConfigEntry("ep.context_file_path", writeContext.absolutePath)
                }
            }
        }
        val loadStart = System.nanoTime()
        env.createSession(path, options).use { session ->
            val loadMs = (System.nanoTime() - loadStart) / 1_000_000
            val inputs = spec.inputs(session)
            val tensors = inputs.mapValues { (_, v) -> OnnxTensor.createTensor(env, FloatBuffer.wrap(v.second), v.first) }
            try {
                val outputs = session.run(tensors).use { r -> OrtValueUtils.extractFloatOutputs(r).map { it.data } }
                val runs = List(if (spec.file == ModelCatalog.FRAN || spec.file == ModelCatalog.GFPGAN) 3 else 5) {
                    val t = System.nanoTime()
                    session.run(tensors).close()
                    (System.nanoTime() - t) / 1_000_000
                }.sorted()
                return Run(loadMs, runs[runs.size / 2], outputs)
            } finally {
                tensors.values.forEach { it.close() }
            }
        }
    }

    /** Worst-case over outputs: NaN count, max error relative to the CPU range, cosine similarity. */
    private fun compare(cpu: List<FloatArray>, npu: List<FloatArray>): String {
        if (cpu.size != npu.size) return "output count differs (${cpu.size} vs ${npu.size})"
        var nan = 0; var worstRel = 0.0; var worstCos = 1.0
        for ((a, b) in cpu.zip(npu)) {
            if (a.size != b.size) return "output size differs (${a.size} vs ${b.size})"
            var range = 1e-6; var maxErr = 0.0; var dot = 0.0; var na = 0.0; var nb = 0.0
            for (i in a.indices) {
                if (b[i].isNaN() || b[i].isInfinite()) { nan++; continue }
                range = max(range, abs(a[i].toDouble()))
                maxErr = max(maxErr, abs(a[i] - b[i]).toDouble())
                dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]
            }
            worstRel = max(worstRel, maxErr / range)
            worstCos = minOf(worstCos, dot / (sqrt(na) * sqrt(nb) + 1e-12))
        }
        return "nan=%d maxRelErr=%.4f cosine=%.5f".format(nan, worstRel, worstCos)
    }

    private companion object {
        const val TAG = "OMNpu"
    }
}
