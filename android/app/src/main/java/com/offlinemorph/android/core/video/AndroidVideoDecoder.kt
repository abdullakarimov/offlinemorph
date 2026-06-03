package com.offlinemorph.android.core.video

import android.content.Context
import android.graphics.Bitmap
import android.media.Image
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Decodes video frames from a URI using [MediaExtractor] + [MediaCodec] in ByteBuffer mode.
 *
 * Frames are yielded as [IndexedFrame] objects via a suspend callback so the caller can apply
 * per-frame work (e.g. face swap) without buffering the whole video in memory.
 */
class AndroidVideoDecoder(private val context: Context) {

    data class VideoTrackInfo(
        val width: Int,
        val height: Int,
        val fps: Float,
        val durationUs: Long,
        val mimeType: String,
    )

    /** Returns basic metadata about the first video track, or null when none is found. */
    fun getVideoInfo(uri: Uri): VideoTrackInfo? {
        val extractor = MediaExtractor()
        return try {
            extractor.setDataSource(context, uri, null)
            val trackIndex = findVideoTrackIndex(extractor) ?: return null
            val format = extractor.getTrackFormat(trackIndex)
            VideoTrackInfo(
                width = format.getInteger(MediaFormat.KEY_WIDTH),
                height = format.getInteger(MediaFormat.KEY_HEIGHT),
                fps = if (format.containsKey(MediaFormat.KEY_FRAME_RATE))
                    format.getInteger(MediaFormat.KEY_FRAME_RATE).toFloat() else 30f,
                durationUs = if (format.containsKey(MediaFormat.KEY_DURATION))
                    format.getLong(MediaFormat.KEY_DURATION) else 0L,
                mimeType = format.getString(MediaFormat.KEY_MIME) ?: "video/avc",
            )
        } catch (_: Exception) {
            null
        } finally {
            extractor.release()
        }
    }

    /**
     * Decodes all frames from [uri] and calls [onFrame] for each one in display order.
     *
     * Must be called from a coroutine. The function switches to [Dispatchers.Default] internally.
     */
    suspend fun decodeFrames(
        uri: Uri,
        cancellationCheck: () -> Boolean = { false },
        onFrame: suspend (IndexedFrame) -> Unit,
    ) = withContext(Dispatchers.Default) {
        val extractor = MediaExtractor()
        extractor.setDataSource(context, uri, null)
        val videoTrackIndex = findVideoTrackIndex(extractor) ?: run {
            extractor.release(); return@withContext
        }
        extractor.selectTrack(videoTrackIndex)

        val format = extractor.getTrackFormat(videoTrackIndex)
        val mimeType = format.getString(MediaFormat.KEY_MIME) ?: run {
            extractor.release(); return@withContext
        }

        val decoder = MediaCodec.createDecoderByType(mimeType)
        // Configure in ByteBuffer mode (null surface) so getOutputImage() is available.
        decoder.configure(format, null, null, 0)
        decoder.start()

        val bufferInfo = MediaCodec.BufferInfo()
        var frameIndex = 0
        var inputDone = false
        var outputDone = false
        val timeoutUs = 10_000L

        try {
            while (!outputDone && !cancellationCheck()) {
                // Feed compressed data to the decoder.
                if (!inputDone) {
                    val inputBufIndex = decoder.dequeueInputBuffer(timeoutUs)
                    if (inputBufIndex >= 0) {
                        val inputBuf = decoder.getInputBuffer(inputBufIndex)!!
                        val sampleSize = extractor.readSampleData(inputBuf, 0)
                        if (sampleSize < 0) {
                            decoder.queueInputBuffer(
                                inputBufIndex, 0, 0, 0L,
                                MediaCodec.BUFFER_FLAG_END_OF_STREAM,
                            )
                            inputDone = true
                        } else {
                            decoder.queueInputBuffer(
                                inputBufIndex, 0, sampleSize,
                                extractor.sampleTime, 0,
                            )
                            extractor.advance()
                        }
                    }
                }

                // Drain decoded output.
                when (val outputBufIndex = decoder.dequeueOutputBuffer(bufferInfo, timeoutUs)) {
                    MediaCodec.INFO_TRY_AGAIN_LATER -> { /* no output yet, keep looping */ }
                    MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> { /* format change, continue */ }
                    else -> {
                        if (outputBufIndex >= 0) {
                            val isEos = (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0
                            if (bufferInfo.size > 0) {
                                val image: Image? = decoder.getOutputImage(outputBufIndex)
                                if (image != null) {
                                    val bitmap = image.toArgbBitmap()
                                    image.close()
                                    onFrame(
                                        IndexedFrame(
                                            index = frameIndex++,
                                            bitmap = bitmap,
                                            presentationTimeUs = bufferInfo.presentationTimeUs,
                                        )
                                    )
                                }
                            }
                            decoder.releaseOutputBuffer(outputBufIndex, false)
                            if (isEos) outputDone = true
                        }
                    }
                }
            }
        } finally {
            decoder.stop()
            decoder.release()
            extractor.release()
        }
    }

    private fun findVideoTrackIndex(extractor: MediaExtractor): Int? {
        for (i in 0 until extractor.trackCount) {
            val mime = extractor.getTrackFormat(i).getString(MediaFormat.KEY_MIME) ?: continue
            if (mime.startsWith("video/")) return i
        }
        return null
    }
}

// --------------------------------------------------------------------------
// YUV_420_888 → ARGB_8888 Bitmap conversion
// --------------------------------------------------------------------------

/**
 * Converts an [Image] in YUV_420_888 format to an ARGB_8888 [Bitmap].
 *
 * Handles both planar (pixelStride=1) and semi-planar (pixelStride=2) U/V layouts,
 * which covers the most common output formats of Android hardware decoders.
 *
 * BT.601 full-range (JFIF) coefficients are used to match the InsightFace / OpenCV default.
 */
private fun Image.toArgbBitmap(): Bitmap {
    val w = width
    val h = height
    val yBuf = planes[0].buffer
    val uBuf = planes[1].buffer
    val vBuf = planes[2].buffer
    val yRowStride = planes[0].rowStride
    val uvRowStride = planes[1].rowStride
    val uvPixelStride = planes[1].pixelStride

    val argb = IntArray(w * h)
    for (j in 0 until h) {
        for (i in 0 until w) {
            val yVal = yBuf.get(j * yRowStride + i).toInt() and 0xFF
            val uvIdx = (j / 2) * uvRowStride + (i / 2) * uvPixelStride
            val uVal = uBuf.get(uvIdx).toInt() and 0xFF
            val vVal = vBuf.get(uvIdx).toInt() and 0xFF

            // BT.601 full-range YUV → RGB
            val r = (yVal + 1.370705f * (vVal - 128)).toInt().coerceIn(0, 255)
            val g = (yVal - 0.698001f * (vVal - 128) - 0.337633f * (uVal - 128)).toInt().coerceIn(0, 255)
            val b = (yVal + 1.732446f * (uVal - 128)).toInt().coerceIn(0, 255)
            argb[j * w + i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
    }
    return Bitmap.createBitmap(argb, w, h, Bitmap.Config.ARGB_8888)
}
