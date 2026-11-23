package com.offlinemorph.android.core.ml

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession
import android.os.Build
import android.util.Log
import com.offlinemorph.android.feature.flags.FeatureFlags
import com.offlinemorph.android.feature.models.ModelCatalog
import java.io.File
import java.nio.FloatBuffer
import java.util.concurrent.ConcurrentHashMap

/**
 * Hands out ONNX Runtime sessions that are cached for the lifetime of the process.
 *
 * Creating a session is by far the most expensive step in the pipeline: inswapper_128 alone is
 * ~550 MB. Sessions are therefore shared across all factory instances and callers must NOT
 * close them.
 * [OrtSession.run] is thread-safe, so a single session can serve concurrent callers.
 *
 * **NPU.** On Qualcomm Snapdragon devices, models in [NpuPolicy.NPU_MODELS] run on the Hexagon
 * NPU through ONNX Runtime's QNN execution provider (HTP backend, FP16). Compiling a graph for
 * the NPU takes seconds (minutes for FRAN), so it is never done on the caller's path:
 * [NpuWarmup] compiles in the background and caches the compiled graph on disk; until a model's
 * cache exists it runs on the CPU, and once it appears a loaded CPU session is upgraded in
 * place. Loading a cached graph takes ~100 ms. Any NPU failure falls back to the CPU.
 */
class OrtSessionFactory {
    private val environment: OrtEnvironment by lazy {
        OrtEnvironment.getEnvironment()
    }

    /** Returns the cached session for [modelFile], loading it on first use. Do not close it. */
    fun session(modelFile: File): OrtSession {
        require(modelFile.exists()) { "Model file does not exist: ${modelFile.absolutePath}" }
        val key = CacheKey(modelFile.absolutePath, modelFile.lastModified(), modelFile.length())
        sessions[key]?.let { return it }
        return synchronized(creationLock) {
            sessions[key] ?: createSession(modelFile).also { created ->
                // A reinstalled model gets a new key; drop the stale session for the same path.
                sessions.keys.filter { it.path == key.path }.forEach { stale ->
                    sessions.remove(stale)?.close()
                }
                sessions[key] = created
            }
        }
    }

    /** Where [modelFile] actually runs once loaded ("NPU" / "CPU"), or null if not loaded yet. */
    fun backendOf(modelFile: File): String? = backends[modelFile.absolutePath]

    private fun createSession(modelFile: File): OrtSession {
        val start = System.nanoTime()
        val npuSession = if (NpuPolicy.shouldTryNpu(modelFile.name)) {
            if (contextFile(modelFile).isFile) {
                runCatching { loadNpuSession(modelFile) }
                    .onFailure { Log.w(TAG, "NPU session for ${modelFile.name} failed; using CPU", it) }
                    .getOrNull()
            } else {
                NpuWarmup.start(modelFile.absoluteFile.parentFile ?: modelFile)
                null
            }
        } else {
            null
        }
        val session = npuSession ?: environment.createSession(modelFile.absolutePath, baseOptions())
        backends[modelFile.absolutePath] = if (npuSession != null) "NPU" else "CPU"
        Log.d(TAG, "Loaded ${modelFile.name} on ${backends[modelFile.absolutePath]} in " +
            "${(System.nanoTime() - start) / 1_000_000} ms")
        return session
    }

    /**
     * CPU execution provider settings shared by every session.
     *
     * Measured on a Galaxy S25 Ultra (Snapdragon 8 Elite) with PipelineOnDeviceTest and
     * NpuBenchmarkTest — re-measure before changing:
     * - NNAPI (ORT 1.18) is 4–5× slower than CPU for inswapper_128 and GFPGAN, and no faster for
     *   the detector or recogniser. XNNPACK aborts on det_10g.
     * - The symbolic dimensions are pinned so every graph has static shapes; the QNN EP rejects
     *   dynamic shapes and would otherwise leave the whole model on the CPU.
     */
    private fun baseOptions() = OrtSession.SessionOptions().apply {
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        setSymbolicDimensionValue("batch", 1L) // fran
        setSymbolicDimensionValue("None", 1L) // w600k_r50, genderage, 2d106det
        setSymbolicDimensionValue("?", 640L) // det_10g input height and width
    }

    /** NPU session from the cached compiled graph of [modelFile]; deletes an unusable cache. */
    private fun loadNpuSession(modelFile: File): OrtSession {
        val context = contextFile(modelFile)
        return runCatching { environment.createSession(context.absolutePath, npuOptions(writeContextTo = null)) }
            .onFailure { context.delete() }
            .getOrThrow()
    }

    /**
     * Compiles [modelFile] for the NPU and writes the compiled graph to its cache file. Slow
     * (seconds to minutes); called only from [NpuWarmup]'s background thread.
     */
    internal fun compileNpuContext(modelFile: File) {
        val context = contextFile(modelFile)
        context.parentFile?.mkdirs()
        // Drop caches compiled from older versions of this model.
        context.parentFile?.listFiles { f -> f.name.startsWith(modelFile.nameWithoutExtension + "-") && f != context }
            ?.forEach { it.delete() }
        val partial = File(context.path + ".tmp").apply { delete() }
        environment.createSession(modelFile.absolutePath, npuOptions(writeContextTo = partial)).close()
        check(partial.isFile && partial.length() > 0) { "QNN wrote no context for ${modelFile.name}" }
        check(partial.renameTo(context)) { "Could not move ${partial.name} into place" }
    }

    /**
     * Replaces an already-loaded CPU session of [modelFile] with an NPU session from the fresh
     * cache. The CPU session is closed only after [RETIRE_DELAY_MS], so runs that already
     * fetched it can finish.
     */
    internal fun upgradeToNpu(modelFile: File) {
        val key = CacheKey(modelFile.absolutePath, modelFile.lastModified(), modelFile.length())
        synchronized(creationLock) {
            val old = sessions[key] ?: return // not loaded yet: the next load picks the NPU
            if (backends[modelFile.absolutePath] == "NPU") return
            val npu = runCatching { loadNpuSession(modelFile) }
                .onFailure { Log.w(TAG, "NPU upgrade of ${modelFile.name} failed; staying on CPU", it) }
                .getOrNull() ?: return
            sessions[key] = npu
            backends[modelFile.absolutePath] = "NPU"
            Log.d(TAG, "Upgraded ${modelFile.name} to NPU")
            retireLater(old)
        }
    }

    private fun retireLater(session: OrtSession) {
        Thread({
            runCatching { Thread.sleep(RETIRE_DELAY_MS) }
            runCatching { session.close() }
        }, "ort-retire").apply { isDaemon = true }.start()
    }

    private fun npuOptions(writeContextTo: File?) = baseOptions().apply {
        addQnn(
            mapOf(
                "backend_path" to "libQnnHtp.so",
                "htp_performance_mode" to "high_performance",
                "enable_htp_fp16_precision" to "1",
                "htp_graph_finalization_optimization_mode" to "3",
            ),
        )
        if (writeContextTo != null) {
            addConfigEntry("ep.context_enable", "1")
            addConfigEntry("ep.context_embed_mode", "1")
            addConfigEntry("ep.context_file_path", writeContextTo.absolutePath)
        }
    }

    /** A new, uncached CPU session for [modelFile], e.g. to cross-check NPU output. Caller closes it. */
    fun cpuSession(modelFile: File): OrtSession = environment.createSession(modelFile.absolutePath, baseOptions())

    /** True once [markNpuValidated] has recorded that this model's NPU output matched the CPU on real data. */
    fun isNpuValidated(modelFile: File): Boolean = File(contextFile(modelFile).parentFile, "${modelFile.name}.validated").exists()

    fun markNpuValidated(modelFile: File) {
        File(contextFile(modelFile).parentFile, "${modelFile.name}.validated").apply { parentFile?.mkdirs(); writeText("1") }
    }

    /**
     * Moves [modelFile] back to the CPU for good (its NPU output did not match): deletes the
     * compiled graph, marks the model failed so it is never recompiled, and swaps the session.
     */
    fun demoteToCpu(modelFile: File) {
        val context = contextFile(modelFile)
        File(context.parentFile, "${modelFile.name}.failed").apply { parentFile?.mkdirs(); writeText("output mismatch") }
        context.delete()
        val key = CacheKey(modelFile.absolutePath, modelFile.lastModified(), modelFile.length())
        synchronized(creationLock) {
            val old = sessions[key]
            sessions[key] = cpuSession(modelFile)
            backends[modelFile.absolutePath] = "CPU"
            old?.let(::retireLater)
        }
        Log.w(TAG, "${modelFile.name} demoted to CPU: NPU output did not match")
    }

    /** True when [modelFile]'s compiled NPU graph is cached. */
    fun hasNpuContext(modelFile: File): Boolean = contextFile(modelFile).isFile

    /**
     * Compiled-graph cache next to the models directory, keyed by model size and mtime so a
     * reinstalled model is recompiled. Lives in `files/qnn_cache/`, outside `files/models/`.
     */
    internal fun contextFile(modelFile: File): File {
        val dir = File(modelFile.absoluteFile.parentFile?.parentFile, "qnn_cache")
        return File(dir, "${modelFile.nameWithoutExtension}-${modelFile.length()}-${modelFile.lastModified()}_ctx.onnx")
    }

    fun createFloatTensor(data: FloatArray, shape: LongArray): OnnxTensor {
        return OnnxTensor.createTensor(environment, FloatBuffer.wrap(data), shape)
    }

    private data class CacheKey(
        val path: String,
        val lastModified: Long,
        val length: Long,
    )

    companion object {
        private const val TAG = "OrtSessionFactory"
        private val sessions = ConcurrentHashMap<CacheKey, OrtSession>()
        private val backends = ConcurrentHashMap<String, String>()
        private val creationLock = Any()

        /** How long a replaced CPU session stays open for in-flight runs (FRAN on CPU: ~40 s/face). */
        private const val RETIRE_DELAY_MS = 5 * 60 * 1000L
    }
}

/**
 * Which models may run on the Hexagon NPU (QNN HTP, FP16).
 *
 * Measured with NpuBenchmarkTest on a Galaxy S25 Ultra (SM8750, HTP v79), CPU → NPU median
 * latency, with NPU output compared against the CPU output on identical inputs:
 *
 * | Model             | CPU      | NPU     | NPU vs CPU                         |
 * |-------------------|----------|---------|------------------------------------|
 * | det_10g           | 242 ms   | 2 ms    | cosine 0.986                       |
 * | w600k_r50         | 97 ms    | 1 ms    | cosine 1.00000                     |
 * | face_parsing      | 187 ms   | 6 ms    | cosine 1.00000                     |
 * | genderage         | <1 ms    | <1 ms   | cosine 1.00000                     |
 * | 2d106det          | 2 ms     | <1 ms   | cosine 1.00000                     |
 * | fran (1024²)      | 37.6 s   | 392 ms  | cosine 1.00000 (compile: 8–12 min) |
 * | GFPGANv1.4        | 2468 ms  | 254 ms  | **NaN / overflow** — CPU only      |
 * | inswapper_128     | 2.3 s    | —       | **FP16 overflow; crashes** — CPU   |
 *
 * inswapper and GFPGAN overflow FP16 (inswapper's AdaIN blocks square their activations), and
 * a native crash cannot be caught and retried on the CPU, so they are excluded up front rather
 * than tried. Running them on the NPU needs INT8/INT16-quantised (QDQ) or FP16-safe exports.
 */
object NpuPolicy {
    val NPU_MODELS = setOf(
        ModelCatalog.DETECTOR,
        ModelCatalog.RECOGNIZER,
        ModelCatalog.GENDERAGE,
        ModelCatalog.LANDMARKS_2D,
        ModelCatalog.FACE_PARSING,
        ModelCatalog.FRAN,
    )

    /** Qualcomm SoC with the FastRPC library the QNN HTP backend needs. */
    val isQualcommNpuDevice: Boolean by lazy {
        val qualcomm = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            Build.SOC_MANUFACTURER.equals("QTI", ignoreCase = true)
        } else {
            Build.HARDWARE.startsWith("qcom", ignoreCase = true)
        }
        qualcomm && listOf("/vendor/lib64/libcdsprpc.so", "/system/vendor/lib64/libcdsprpc.so").any { File(it).exists() }
    }

    /** Detected 5-point landmarks may differ from the CPU's by at most this share of the eye distance. */
    const val DETECTOR_TOLERANCE = 0.03f

    fun shouldTryNpu(modelFileName: String): Boolean =
        FeatureFlags.npuAccelerationEnabled && isQualcommNpuDevice && modelFileName in NPU_MODELS
}

/**
 * Background compilation of NPU graphs. One low-priority thread compiles, in [NpuPolicy.NPU_MODELS]
 * order, every installed model without a cached graph, then upgrades loaded CPU sessions.
 *
 * Crash guard: a `.compiling` marker holding an attempt count is written before each compile and
 * removed after. A process that dies mid-compile leaves it behind. That is usually harmless (the
 * app was closed, updated or killed in the background; FRAN takes ~12 minutes), so the compile
 * is retried, but after [MAX_ATTEMPTS] interrupted attempts the model is assumed to crash the
 * QNN compiler, marked `.failed`, and left on the CPU for good.
 */
object NpuWarmup {
    private const val TAG = "NpuWarmup"

    sealed interface Status {
        data object Idle : Status
        data class Compiling(val modelName: String, val remaining: Int) : Status
        data object Done : Status
    }

    private val _status = kotlinx.coroutines.flow.MutableStateFlow<Status>(Status.Idle)
    val status: kotlinx.coroutines.flow.StateFlow<Status> = _status

    @Volatile private var running = false
    @Volatile private var rescan = false
    private val factory = OrtSessionFactory()

    /** Starts (or re-triggers) background compilation for models in [modelsDirectory]. */
    fun start(modelsDirectory: File) {
        if (!NpuPolicy.isQualcommNpuDevice || !FeatureFlags.npuAccelerationEnabled) return
        synchronized(this) {
            if (running) { rescan = true; return }
            running = true
        }
        Thread({ loop(modelsDirectory) }, "npu-warmup").apply {
            priority = Thread.MIN_PRIORITY
            isDaemon = true
            start()
        }
    }

    private fun pending(modelsDirectory: File): List<File> =
        NpuPolicy.NPU_MODELS.map { File(modelsDirectory, it) }
            .filter { it.isFile && !factory.hasNpuContext(it) && !marker(it, "failed").exists() }

    private const val MAX_ATTEMPTS = 5

    private fun attempts(marker: File): Int =
        if (marker.exists()) marker.readText().trim().toIntOrNull() ?: 1 else 0

    private fun marker(model: File, kind: String) = File(factory.contextFile(model).parentFile, "${model.name}.$kind")

    private fun loop(modelsDirectory: File) {
        try {
            // Markers left by previous runs are interrupted compiles; give up after MAX_ATTEMPTS.
            NpuPolicy.NPU_MODELS.map { File(modelsDirectory, it) }.forEach { model ->
                val compiling = marker(model, "compiling")
                if (attempts(compiling) >= MAX_ATTEMPTS) {
                    Log.w(TAG, "${model.name}: $MAX_ATTEMPTS NPU compiles were interrupted; keeping it on the CPU")
                    compiling.renameTo(marker(model, "failed"))
                }
            }
            do {
                rescan = false
                var queue = pending(modelsDirectory)
                while (queue.isNotEmpty()) {
                    val model = queue.first()
                    _status.value = Status.Compiling(model.name, queue.size)
                    val compiling = marker(model, "compiling")
                    compiling.parentFile?.mkdirs()
                    compiling.writeText((attempts(compiling) + 1).toString())
                    val start = System.nanoTime()
                    runCatching { factory.compileNpuContext(model) }
                        .onSuccess {
                            Log.i(TAG, "Compiled ${model.name} for the NPU in ${(System.nanoTime() - start) / 1_000_000} ms")
                            compiling.delete()
                            factory.upgradeToNpu(model)
                        }
                        .onFailure {
                            Log.w(TAG, "NPU compile of ${model.name} failed; keeping it on the CPU", it)
                            compiling.renameTo(marker(model, "failed"))
                        }
                    queue = pending(modelsDirectory)
                }
            } while (rescan)
            _status.value = Status.Done
        } finally {
            synchronized(this) { running = false }
        }
    }
}
