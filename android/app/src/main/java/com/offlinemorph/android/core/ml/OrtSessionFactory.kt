package com.offlinemorph.android.core.ml

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession
import android.util.Log
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

    /**
     * CPU execution provider only. Measured on a Galaxy S25 Ultra (Snapdragon 8 Elite) with
     * PipelineOnDeviceTest.executionProviderBenchmark — re-measure before changing:
     * - NNAPI (ORT 1.18) is 4–5× slower than CPU for inswapper_128 (10.2 s vs 2.3 s) and GFPGAN
     *   (19 s vs 3.7 s), and no faster for the detector or recogniser. XNNPACK aborts on det_10g.
     * - QNN HTP (onnxruntime-android-qnn 1.29 plus `<uses-native-library libcdsprpc.so>` in the
     *   manifest and extracted native libs) runs inswapper in 42 ms and GFPGAN in 0.31 s, but
     *   the NPU computes floats in FP16 and both models overflow: inswapper returns a near-black
     *   image and GFPGAN returns NaN. Using the NPU needs INT8/INT16 quantised (QDQ) or
     *   FP16-safe versions of these models. The QNN GPU backend silently falls back to CPU.
     */
    private fun createSession(modelFile: File): OrtSession {
        val start = System.nanoTime()
        val options = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        return environment.createSession(modelFile.absolutePath, options).also {
            Log.d(TAG, "Loaded ${modelFile.name} in ${(System.nanoTime() - start) / 1_000_000} ms")
        }
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
        private val creationLock = Any()
    }
}
