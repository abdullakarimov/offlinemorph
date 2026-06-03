package com.offlinemorph.android.core.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.File
import java.nio.FloatBuffer

/**
 * Optional post-swap face restoration enhancer.
 *
 * Accepts the raw 128×128 CHW RGB [0, 1] float tensor produced by the inswapper and upscales
 * it to a sharp 512×512 output via a lightweight mobile face super-resolution ONNX model
 * (e.g. a mobile-optimised GFPGAN 1.4 or similar face patch network).
 *
 * **Expected ONNX model contract**
 * ```
 *   input:  name "input",  shape [1, 3, 128, 128], values in [-1, 1]
 *   output: name "output", shape [1, 3, 512, 512], values in [-1, 1]
 * ```
 * Input/output normalization is handled internally:
 * - Before inference:  [0, 1] → [-1, 1]   (`pixel * 2 − 1`)
 * - After inference:   [-1, 1] → [0, 1]   (`(pixel + 1) / 2`, clamped)
 *
 * If the model file is absent or inference fails, [enhanceToBitmap] returns `null` and the
 * caller is expected to fall back to the standard Lanczos4 path in [FaceAlignmentOps].
 *
 * The ONNX session is created lazily on the first [enhanceToBitmap] call and cached for
 * subsequent swaps. Call [close] when the owning engine is destroyed to free native memory.
 */
class OnDeviceFaceEnhancer(
    private val modelsDirectory: File,
) {
    companion object {
        /** Model filename expected in the app-private models directory. */
        const val MODEL_FILE  = "GFPGANv1.4.onnx"
        /** GFPGANv1.4 ONNX model expects a 512×512 CHW input (same as Python preprocessing). */
        const val INPUT_SIZE  = 512
        const val OUTPUT_SIZE = 512
        private const val RAW_PATCH_SIZE = 128
        private const val TAG = "FaceEnhancer"
    }

    private val env: OrtEnvironment by lazy { OrtEnvironment.getEnvironment() }

    @Volatile private var session: OrtSession? = null

    // ── Public API ─────────────────────────────────────────────────────────────

    /** Returns `true` when the enhancer model file is present in app-private storage. */
    fun isAvailable(): Boolean = File(modelsDirectory, MODEL_FILE).isFile

    /**
     * Runs face-restoration inference on [inputCHW] and decodes the result to a [Bitmap].
     *
     * @param inputCHW 128×128 CHW RGB float array in [0, 1] — matches the raw inswapper output.
     * @return a 512×512 [Bitmap.Config.ARGB_8888] bitmap on success, or `null` on model absence
     *         or inference failure. A `null` result signals the caller to fall back to Lanczos4.
     */
    fun enhanceToBitmap(inputCHW: FloatArray): Bitmap? {
        val outputCHW = runEnhance(inputCHW) ?: return null
        return decodeCHWtoBitmap(outputCHW, OUTPUT_SIZE)
    }

    /**
     * Releases the underlying ONNX session and frees all native C++ heap memory.
     *
     * Should be called from the owning component's `onCleared()` / `onDestroy()` to prevent
     * memory leaks across swap sessions.
     */
    fun close() {
        session?.close()
        session = null
    }

    // ── Inference ──────────────────────────────────────────────────────────────

    private fun runEnhance(inputCHW: FloatArray): FloatArray? {
        if (!isAvailable()) {
            Log.d(TAG, "Model file absent — skipping enhancement.")
            return null
        }
        return runCatching {
            val sess = getOrCreateSession() ?: return null

            val rawPlane = RAW_PATCH_SIZE * RAW_PATCH_SIZE
            require(inputCHW.size == rawPlane * 3) {
                "Expected ${rawPlane * 3} floats (128×128 CHW), got ${inputCHW.size}"
            }

            // GFPGANv1.4 expects 512×512 input. Upscale the 128×128 inswapper patch via
            // OpenCV INTER_CUBIC, matching Python: cv2.resize(frame, (512,512), cv2.INTER_CUBIC).
            val upscaled = upscale128to512(inputCHW) ?: return null

            // Normalise [0, 1] → [-1, 1]: matches Python (frame - 0.5) / 0.5.
            val normalized = FloatArray(upscaled.size) { i -> upscaled[i] * 2.0f - 1.0f }
            val inputShape = longArrayOf(1L, 3L, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())

            OnnxTensor.createTensor(env, FloatBuffer.wrap(normalized), inputShape).use { inputTensor ->
                val inputName = sess.inputNames.firstOrNull() ?: "input"
                sess.run(mapOf(inputName to inputTensor)).use { result ->
                    // Select the output tensor that best matches the expected 512×512×3 flat size.
                    val outputs = OrtValueUtils.extractFloatOutputs(result)
                    val expectedFlat = OUTPUT_SIZE * OUTPUT_SIZE * 3
                    val best = outputs.firstOrNull { it.data.size == expectedFlat }
                        ?: outputs.maxByOrNull { it.data.size }
                        ?: return null

                    if (best.data.size != expectedFlat) {
                        Log.w(TAG, "Unexpected output size ${best.data.size}, expected $expectedFlat")
                        return null
                    }
                    // Denormalise [-1, 1] → [0, 1] and clamp to valid pixel range.
                    FloatArray(best.data.size) { i ->
                        ((best.data[i] + 1.0f) / 2.0f).coerceIn(0f, 1f)
                    }
                }
            }
        }.getOrElse { e ->
            Log.e(TAG, "Enhancement inference failed", e)
            null
        }
    }

    // ── Preprocessing ──────────────────────────────────────────────────────────

    /**
     * Decodes the 128×128 CHW float patch and resizes it to 512×512 CHW via OpenCV INTER_CUBIC,
     * matching Python `cv2.resize(frame, (512, 512), cv2.INTER_CUBIC)` in the GFPGAN pipeline.
     * All native Mat allocations are released before returning.
     */
    private fun upscale128to512(chw128: FloatArray): FloatArray? {
        val plane128 = RAW_PATCH_SIZE * RAW_PATCH_SIZE
        val pixels128 = IntArray(plane128) { i ->
            val r = (chw128[i]                * 255f).toInt().coerceIn(0, 255)
            val g = (chw128[plane128 + i]     * 255f).toInt().coerceIn(0, 255)
            val b = (chw128[plane128 * 2 + i] * 255f).toInt().coerceIn(0, 255)
            (255 shl 24) or (r shl 16) or (g shl 8) or b
        }
        val bmp128 = Bitmap.createBitmap(pixels128, RAW_PATCH_SIZE, RAW_PATCH_SIZE, Bitmap.Config.ARGB_8888)

        val srcMat = Mat()
        Utils.bitmapToMat(bmp128, srcMat)
        bmp128.recycle()
        val dstMat = Mat()
        Imgproc.resize(srcMat, dstMat, Size(INPUT_SIZE.toDouble(), INPUT_SIZE.toDouble()), 0.0, 0.0, Imgproc.INTER_CUBIC)
        srcMat.release()

        val bmp512 = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(dstMat, bmp512)
        dstMat.release()

        val plane512 = INPUT_SIZE * INPUT_SIZE
        val pixels512 = IntArray(plane512)
        bmp512.getPixels(pixels512, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        bmp512.recycle()

        return FloatArray(plane512 * 3) { idx ->
            val plane = idx / plane512
            val i = idx % plane512
            val pixel = pixels512[i]
            when (plane) {
                0 -> ((pixel shr 16) and 0xFF) / 255f  // R
                1 -> ((pixel shr 8)  and 0xFF) / 255f  // G
                else -> (pixel and 0xFF)        / 255f  // B
            }
        }
    }

    // ── Session management ─────────────────────────────────────────────────────

    private fun getOrCreateSession(): OrtSession? {
        session?.let { return it }
        return synchronized(this) {
            session ?: runCatching {
                val modelFile = File(modelsDirectory, MODEL_FILE)
                // Request NNAPI execution provider to offload to GPU/DSP on supported hardware.
                // Falls back to CPU automatically when NNAPI is unavailable.
                val options = OrtSession.SessionOptions().apply {
                    addNnapi()
                }
                env.createSession(modelFile.absolutePath, options).also { sess ->
                    session = sess
                    Log.d(TAG, "Enhancer session ready — inputs=${sess.inputNames} outputs=${sess.outputNames}")
                }
            }.getOrElse { e ->
                Log.e(TAG, "Failed to create enhancer session", e)
                null
            }
        }
    }

    // ── Bitmap decode ──────────────────────────────────────────────────────────

    /**
     * Decodes a CHW RGB float array in [0, 1] to an [Bitmap.Config.ARGB_8888] bitmap.
     * All intermediate allocations stay on the JVM heap; no native OpenCV Mat is allocated.
     */
    private fun decodeCHWtoBitmap(chw: FloatArray, size: Int): Bitmap {
        val plane = size * size
        val pixels = IntArray(plane)
        var i = 0
        while (i < plane) {
            val r = (chw[i]             * 255f).toInt().coerceIn(0, 255)
            val g = (chw[plane + i]     * 255f).toInt().coerceIn(0, 255)
            val b = (chw[plane * 2 + i] * 255f).toInt().coerceIn(0, 255)
            pixels[i] = (255 shl 24) or (r shl 16) or (g shl 8) or b
            i++
        }
        return Bitmap.createBitmap(pixels, size, size, Bitmap.Config.ARGB_8888)
    }
}
