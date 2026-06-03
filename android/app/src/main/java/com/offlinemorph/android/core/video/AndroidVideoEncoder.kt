package com.offlinemorph.android.core.video

import android.graphics.Bitmap
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMuxer
import java.io.File
import java.nio.ByteBuffer

/**
 * Encodes a sequence of [Bitmap] frames into an H.264 MP4 file using [MediaCodec] in ByteBuffer
 * input mode.  Each frame is converted from ARGB to YUV 420 SemiPlanar (NV12) before being fed
 * to the encoder so presentation timestamps set by the caller are preserved exactly.
 *
 * Usage:
 * ```
 * val encoder = AndroidVideoEncoder(outputFile, width, height, fps)
 * encoder.start()
 * frames.forEach { (bitmap, pts) -> encoder.encodeFrame(bitmap, pts) }
 * encoder.finish()
 * ```
 *
 * After [finish], [outputVideoFile] contains a video-only MP4.  Call [remuxWithAudio] separately
 * to attach an audio track from the original source video.
 */
class AndroidVideoEncoder(
    val outputVideoFile: File,
    private val width: Int,
    private val height: Int,
    private val fps: Int,
    private val bitrate: Int = 4_000_000,
) {
    private var encoder: MediaCodec? = null
    private var muxer: MediaMuxer? = null
    private var videoTrackIndex = -1
    private var muxerStarted = false
    private val bufferInfo = MediaCodec.BufferInfo()

    fun start() {
        val format = MediaFormat.createVideoFormat("video/avc", width, height).apply {
            setInteger(
                MediaFormat.KEY_COLOR_FORMAT,
                MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar,
            )
            setInteger(MediaFormat.KEY_BIT_RATE, bitrate)
            setInteger(MediaFormat.KEY_FRAME_RATE, fps)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
        }
        encoder = MediaCodec.createEncoderByType("video/avc").also { enc ->
            enc.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            enc.start()
        }
        muxer = MediaMuxer(outputVideoFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
    }

    /**
     * Encodes one [bitmap] with the given [presentationTimeUs] (from the decoder).
     * The bitmap is automatically scaled to [width]×[height] when needed.
     */
    fun encodeFrame(bitmap: Bitmap, presentationTimeUs: Long) {
        val enc = encoder ?: return

        // Resize if the swap engine produced a differently-sized output.
        val src = if (bitmap.width == width && bitmap.height == height) bitmap
                  else Bitmap.createScaledBitmap(bitmap, width, height, true)

        val inputIndex = enc.dequeueInputBuffer(10_000L)
        if (inputIndex >= 0) {
            val inputBuf = enc.getInputBuffer(inputIndex)!!
            inputBuf.clear()
            inputBuf.put(bitmapToNv12(src))
            enc.queueInputBuffer(inputIndex, 0, inputBuf.position(), presentationTimeUs, 0)
        }
        drainEncoder(endOfStream = false)
    }

    /** Signals end-of-stream, flushes all remaining encoded frames, and releases resources. */
    fun finish() {
        val enc = encoder ?: return
        // Signal EOS via an empty input buffer.
        val inputIndex = enc.dequeueInputBuffer(10_000L)
        if (inputIndex >= 0) {
            enc.queueInputBuffer(inputIndex, 0, 0, 0L, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
        }
        drainEncoder(endOfStream = true)
        enc.stop()
        enc.release()
        encoder = null
        muxer?.stop()
        muxer?.release()
        muxer = null
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    private fun drainEncoder(endOfStream: Boolean) {
        val enc = encoder ?: return
        val mux = muxer ?: return
        val timeout = if (endOfStream) 10_000L else 0L

        while (true) {
            val index = enc.dequeueOutputBuffer(bufferInfo, timeout)
            when {
                index == MediaCodec.INFO_TRY_AGAIN_LATER -> {
                    if (!endOfStream) break
                }
                index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    videoTrackIndex = mux.addTrack(enc.outputFormat)
                    mux.start()
                    muxerStarted = true
                }
                index >= 0 -> {
                    val codecConfigFlag = MediaCodec.BUFFER_FLAG_CODEC_CONFIG
                    val eosFlag = MediaCodec.BUFFER_FLAG_END_OF_STREAM
                    if ((bufferInfo.flags and codecConfigFlag) == 0 &&
                        bufferInfo.size > 0 &&
                        muxerStarted
                    ) {
                        val outputBuffer = enc.getOutputBuffer(index)!!
                        outputBuffer.position(bufferInfo.offset)
                        outputBuffer.limit(bufferInfo.offset + bufferInfo.size)
                        mux.writeSampleData(videoTrackIndex, outputBuffer, bufferInfo)
                    }
                    enc.releaseOutputBuffer(index, false)
                    if ((bufferInfo.flags and eosFlag) != 0) return
                }
            }
        }
    }

    /** ARGB_8888 Bitmap → YUV 420 SemiPlanar (NV12) byte array. */
    private fun bitmapToNv12(bitmap: Bitmap): ByteArray {
        val w = bitmap.width
        val h = bitmap.height
        val argb = IntArray(w * h)
        bitmap.getPixels(argb, 0, w, 0, 0, w, h)

        // Y plane: w×h bytes; UV plane: (w/2)×(h/2) interleaved U,V pairs = w×h/2 bytes.
        val nv12 = ByteArray(w * h + (w * h / 2))
        val uvOffset = w * h

        for (j in 0 until h) {
            for (i in 0 until w) {
                val pixel = argb[j * w + i]
                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF

                // BT.601 limited-range (studio swing)
                val y = ((66 * r + 129 * g + 25 * b + 128) shr 8) + 16
                nv12[j * w + i] = y.coerceIn(16, 235).toByte()

                if (j % 2 == 0 && i % 2 == 0) {
                    val u = ((-38 * r - 74 * g + 112 * b + 128) shr 8) + 128
                    val v = ((112 * r - 94 * g - 18 * b + 128) shr 8) + 128
                    nv12[uvOffset + (j / 2) * w + i]     = u.coerceIn(16, 240).toByte()
                    nv12[uvOffset + (j / 2) * w + i + 1] = v.coerceIn(16, 240).toByte()
                }
            }
        }
        return nv12
    }
}

// --------------------------------------------------------------------------
// Audio remux helper — public top-level function used by OnDeviceVideoSwapEngine
// --------------------------------------------------------------------------

/**
 * Remuxes a video-only [tempVideoFile] together with the audio track of [audioSourceExtractor]
 * into the final [outputFile].
 *
 * [audioTrackIndex] must be already selected on [audioSourceExtractor] before calling this.
 */
fun remuxVideoWithAudio(
    tempVideoFile: File,
    audioSourceExtractor: MediaExtractor,
    audioTrackIndex: Int,
    outputFile: File,
) {
    val videoExtractor = MediaExtractor()
    videoExtractor.setDataSource(tempVideoFile.absolutePath)
    val videoTrackIdx = (0 until videoExtractor.trackCount).firstOrNull { i ->
        val mime = videoExtractor.getTrackFormat(i).getString(MediaFormat.KEY_MIME) ?: ""
        mime.startsWith("video/")
    } ?: run {
        videoExtractor.release()
        return
    }
    videoExtractor.selectTrack(videoTrackIdx)
    audioSourceExtractor.selectTrack(audioTrackIndex)

    val muxer = MediaMuxer(outputFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
    val muxVideoTrack = muxer.addTrack(videoExtractor.getTrackFormat(videoTrackIdx))
    val muxAudioTrack = muxer.addTrack(audioSourceExtractor.getTrackFormat(audioTrackIndex))
    muxer.start()

    val buf = ByteBuffer.allocate(512 * 1024)
    val info = MediaCodec.BufferInfo()

    // Copy video samples.
    while (true) {
        buf.clear()
        val size = videoExtractor.readSampleData(buf, 0)
        if (size < 0) break
        info.presentationTimeUs = videoExtractor.sampleTime
        info.size = size
        info.flags = videoExtractor.sampleFlags
        info.offset = 0
        muxer.writeSampleData(muxVideoTrack, buf, info)
        videoExtractor.advance()
    }

    // Copy audio samples (truncated to video duration if longer).
    val videoDurationUs = (0 until videoExtractor.trackCount)
        .firstOrNull()
        ?.let { videoExtractor.getTrackFormat(videoTrackIdx) }
        ?.let { if (it.containsKey(MediaFormat.KEY_DURATION)) it.getLong(MediaFormat.KEY_DURATION) else Long.MAX_VALUE }
        ?: Long.MAX_VALUE
    while (true) {
        buf.clear()
        val size = audioSourceExtractor.readSampleData(buf, 0)
        if (size < 0) break
        val pts = audioSourceExtractor.sampleTime
        if (pts > videoDurationUs) break
        info.presentationTimeUs = pts
        info.size = size
        info.flags = audioSourceExtractor.sampleFlags
        info.offset = 0
        muxer.writeSampleData(muxAudioTrack, buf, info)
        audioSourceExtractor.advance()
    }

    muxer.stop()
    muxer.release()
    videoExtractor.release()
}
