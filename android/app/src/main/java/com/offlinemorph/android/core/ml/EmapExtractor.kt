package com.offlinemorph.android.core.ml

import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * Extracts the emap embedding projection matrix from inswapper_128.onnx.
 *
 * The swap model stores the embedding map as its last graph initializer.
 * Python equivalent:
 *   graph = onnx.load(path).graph
 *   emap = onnx.numpy_helper.to_array(graph.initializer[-1])
 *
 * A memory-mapped buffer is used so the full ~500 MB model is never loaded into RAM;
 * the OS pages in only what the parser touches.  The result is written to a small
 * sidecar file (inswapper_emap.bin) so extraction only runs once per model install.
 */
object EmapExtractor {

    private const val SIDECAR_NAME = "inswapper_emap.bin"

    // ── Public API ────────────────────────────────────────────────────────────

    /** Returns the emap float array, using the on-disk cache when available. */
    fun getEmap(modelFile: File): FloatArray {
        val sidecar = File(modelFile.parentFile, SIDECAR_NAME)
        if (sidecar.isFile && sidecar.length() > 0) {
            val cached = readBinaryFloats(sidecar)
            // Reject a sidecar written with wrong byte-order (NaN values) and re-extract.
            if (cached.none { it.isNaN() }) return cached
            sidecar.delete()
        }
        val emap = extractFromModel(modelFile)
        runCatching { writeBinaryFloats(sidecar, emap) }
        return emap
    }

    // ── Cache helpers ─────────────────────────────────────────────────────────

    private fun readBinaryFloats(file: File): FloatArray {
        val bytes = file.readBytes()
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val out = FloatArray(bytes.size / 4)
        buf.asFloatBuffer().get(out)
        return out
    }

    private fun writeBinaryFloats(file: File, data: FloatArray) {
        val buf = ByteBuffer.allocate(data.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        buf.asFloatBuffer().put(data)
        file.writeBytes(buf.array())
    }

    // ── Protobuf navigation ───────────────────────────────────────────────────

    private fun extractFromModel(modelFile: File): FloatArray {
        FileInputStream(modelFile).use { fis ->
            val channel = fis.channel
            val fileSize = channel.size()
            val buf = channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize)
            buf.order(ByteOrder.LITTLE_ENDIAN)

            // ModelProto  →  graph (field 7, wire type 2)
            val graphSlice = findLengthDelimitedField(buf, fieldNumber = 7)
                ?: error("Graph field (7) not found in ${modelFile.name}")

            // GraphProto  →  emap initializer: the unique [512 × 512] TensorProto
            val tensorSlice = findEmapInitializer(graphSlice)
                ?: error("No [512×512] initializer found in graph of ${modelFile.name}")

            return decodeTensorFloats(tensorSlice)
        }
    }

    /**
     * Finds the first length-delimited field with [fieldNumber] in [buf].
     * Returns a slice whose position/limit cover only that field's payload bytes.
     * On success the caller's [buf] is advanced past the field; on failure the
     * buffer position is left unspecified.
     */
    private fun findLengthDelimitedField(buf: ByteBuffer, fieldNumber: Int): ByteBuffer? {
        while (buf.hasRemaining()) {
            val tag = readVarint32(buf)
            val wireType = tag and 7
            val fn = tag ushr 3
            if (fn == fieldNumber && wireType == 2) {
                val length = readVarint32(buf)
                val start = buf.position()
                val slice = buf.duplicate()
                slice.position(start)
                slice.limit(start + length)
                buf.position(start + length)
                return slice
            }
            skipField(buf, wireType)
        }
        return null
    }

    /**
     * Scans all field-5 (TensorProto initializer) entries in the GraphProto and returns
     * the slice for the tensor with dims [512, 512] — the emap projection matrix.
     * Falls back to the last TensorProto with exactly 262144 (512×512) elements if
     * no exact [512, 512] dims match is found.
     */
    private fun findEmapInitializer(graphBuf: ByteBuffer): ByteBuffer? {
        var fallbackSlice: ByteBuffer? = null
        while (graphBuf.hasRemaining()) {
            val tag = readVarint32(graphBuf)
            val wireType = tag and 7
            val fn = tag ushr 3
            if (fn == 5 && wireType == 2) {
                val length = readVarint32(graphBuf)
                val start = graphBuf.position()
                val end = start + length
                val tensorSlice = graphBuf.duplicate().also {
                    it.position(start)
                    it.limit(end)
                }
                graphBuf.position(end)
                val dims = readTensorDims(tensorSlice.duplicate())
                if (dims.size == 2 && dims[0] == 512L && dims[1] == 512L) {
                    return tensorSlice  // Exact match: [512, 512] emap
                }
                if (dims.isNotEmpty() && dims.fold(1L) { acc, d -> acc * d } == 262144L) {
                    fallbackSlice = tensorSlice  // 262144-element fallback
                }
            } else {
                skipField(graphBuf, wireType)
            }
        }
        return fallbackSlice
    }

    /**
     * Reads the dim values (field 1, repeated int64) from a TensorProto.
     * Stops early before reaching float_data / raw_data to avoid scanning large payloads.
     */
    private fun readTensorDims(buf: ByteBuffer): List<Long> {
        val dims = mutableListOf<Long>()
        while (buf.hasRemaining()) {
            val tag = readVarint32(buf)
            val wireType = tag and 7
            val fn = tag ushr 3
            when {
                fn == 1 && wireType == 0 -> dims.add(readVarint64(buf))
                fn == 2 && wireType == 0 -> readVarint64(buf)  // data_type — skip
                fn >= 4                  -> break               // float_data / raw_data — stop
                else                     -> skipField(buf, wireType)
            }
        }
        return dims
    }

    /**
     * Extracts float32 data from a TensorProto slice.
     * Prefers `raw_data` (field 9) over `float_data` (field 4) when both are present.
     *
     * Note: Android's MappedByteBuffer.duplicate() does NOT propagate the byte order of
     * the parent buffer — the duplicate always starts as BIG_ENDIAN.  We therefore force
     * LITTLE_ENDIAN explicitly here before any multi-byte read.
     */
    private fun decodeTensorFloats(buf: ByteBuffer): FloatArray {
        buf.order(ByteOrder.LITTLE_ENDIAN)
        var rawStart = -1
        var rawLength = 0
        var floatData: FloatArray? = null

        while (buf.hasRemaining()) {
            val tag = readVarint32(buf)
            val wireType = tag and 7
            val fn = tag ushr 3
            when {
                fn == 4 && wireType == 2 -> {
                    // float_data: packed IEEE 754 float32 values
                    val len = readVarint32(buf)
                    val arr = FloatArray(len / 4)
                    for (i in arr.indices) arr[i] = buf.float
                    floatData = arr
                }
                fn == 9 && wireType == 2 -> {
                    // raw_data: little-endian float32 bytes
                    rawLength = readVarint32(buf)
                    rawStart = buf.position()
                    buf.position(rawStart + rawLength)
                }
                else -> skipField(buf, wireType)
            }
        }

        if (rawStart >= 0 && rawLength > 0) {
            buf.position(rawStart)
            return FloatArray(rawLength / 4).also { arr ->
                for (i in arr.indices) arr[i] = buf.float
            }
        }

        return floatData ?: error("No float data found in TensorProto")
    }

    // ── Protobuf wire-format primitives ───────────────────────────────────────

    private fun readVarint32(buf: ByteBuffer): Int {
        var result = 0
        var shift = 0
        while (buf.hasRemaining()) {
            val b = buf.get().toInt() and 0xFF
            result = result or ((b and 0x7F) shl shift)
            if (b and 0x80 == 0) break
            shift += 7
        }
        return result
    }

    private fun readVarint64(buf: ByteBuffer): Long {
        var result = 0L
        var shift = 0
        while (buf.hasRemaining()) {
            val b = buf.get().toInt() and 0xFF
            result = result or ((b and 0x7F).toLong() shl shift)
            if (b and 0x80 == 0) break
            shift += 7
        }
        return result
    }

    private fun skipField(buf: ByteBuffer, wireType: Int) {
        when (wireType) {
            0 -> readVarint64(buf)
            1 -> { if (buf.remaining() >= 8) buf.position(buf.position() + 8) }
            2 -> {
                val len = readVarint32(buf)
                if (buf.remaining() >= len) buf.position(buf.position() + len)
            }
            5 -> { if (buf.remaining() >= 4) buf.position(buf.position() + 4) }
            else -> error("Unknown protobuf wire type $wireType at position ${buf.position()}")
        }
    }
}
