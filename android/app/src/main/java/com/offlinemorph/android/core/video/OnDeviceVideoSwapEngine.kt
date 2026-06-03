package com.offlinemorph.android.core.video

import android.content.Context
import android.media.MediaExtractor
import android.media.MediaFormat
import com.offlinemorph.android.core.ml.FaceSwapEngine
import com.offlinemorph.android.core.ml.SwapRequest
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * On-device video face-swap engine.
 *
 * Pipeline (two passes):
 *  1. Decode each frame from [targetVideoUri] → run face swap via [faceSwapEngine] → encode
 *     swapped frames to a temporary video-only MP4 in [outputDir].
 *  2. Remux the temp video with the original audio track from [targetVideoUri] into the final
 *     output MP4.  When no audio track exists the temp file is renamed directly.
 *
 * Memory: only one decoded [android.graphics.Bitmap] plus one swapped [android.graphics.Bitmap]
 * are held in memory at a time.  Decoded bitmaps are recycled after the swap engine returns.
 */
class OnDeviceVideoSwapEngine(
    private val context: Context,
    private val faceSwapEngine: FaceSwapEngine,
    private val outputDir: File,
) : VideoSwapEngine {

    override suspend fun swapVideo(
        request: VideoSwapRequest,
        cancellationCheck: () -> Boolean,
        onProgress: (Int, Int, String) -> Unit,
    ): VideoSwapResult = withContext(Dispatchers.Default) {
        val decoder = AndroidVideoDecoder(context)
        val videoInfo = decoder.getVideoInfo(request.targetVideoUri)
            ?: return@withContext VideoSwapResult(null, "Failed to read video track info.")

        val estimatedFrames = if (videoInfo.durationUs > 0 && videoInfo.fps > 0f)
            ((videoInfo.durationUs / 1_000_000.0) * videoInfo.fps).toInt().coerceAtLeast(1)
        else 0  // unknown

        outputDir.mkdirs()
        val ts = System.currentTimeMillis()
        val tempVideoFile = File(outputDir, "tmp_video_$ts.mp4")
        val outputFile = File(outputDir, "swap_video_$ts.mp4")

        // ── Pass 1: decode → swap → encode ───────────────────────────────────
        val encoder = AndroidVideoEncoder(
            outputVideoFile = tempVideoFile,
            width  = videoInfo.width,
            height = videoInfo.height,
            fps    = videoInfo.fps.toInt().coerceIn(10, 120),
        )
        encoder.start()

        var framesProcessed = 0
        try {
            decoder.decodeFrames(
                uri = request.targetVideoUri,
                cancellationCheck = cancellationCheck,
            ) { frame ->
                val swapRequest = SwapRequest(
                    sourceBitmap    = request.sourceBitmap,
                    targetBitmap    = frame.bitmap,
                    enhancerEnabled = request.enhancerEnabled,
                    targetFaceIndex = request.targetFaceIndex,
                    faceFilterMode  = request.faceFilterMode,
                )
                val result = faceSwapEngine.runSwap(swapRequest)
                val swappedBitmap = result.outputBitmap ?: frame.bitmap
                encoder.encodeFrame(swappedBitmap, frame.presentationTimeUs)

                // Recycle the decoded frame bitmap if we got a fresh swapped copy.
                if (result.outputBitmap != null && !frame.bitmap.isRecycled) {
                    frame.bitmap.recycle()
                }

                framesProcessed++
                val total = if (estimatedFrames > 0) estimatedFrames else framesProcessed
                onProgress(framesProcessed, total, "Frame $framesProcessed / ~$total")
            }
        } finally {
            encoder.finish()
        }

        if (cancellationCheck()) {
            tempVideoFile.delete()
            return@withContext VideoSwapResult(null, "Cancelled after $framesProcessed frames.")
        }

        // ── Pass 2: remux with audio ──────────────────────────────────────────
        val audioExtractor = MediaExtractor()
        audioExtractor.setDataSource(context, request.targetVideoUri, null)
        val audioTrackIndex = (0 until audioExtractor.trackCount).firstOrNull { i ->
            val mime = audioExtractor.getTrackFormat(i).getString(MediaFormat.KEY_MIME) ?: ""
            mime.startsWith("audio/")
        }

        if (audioTrackIndex != null) {
            try {
                remuxVideoWithAudio(tempVideoFile, audioExtractor, audioTrackIndex, outputFile)
            } finally {
                audioExtractor.release()
                tempVideoFile.delete()
            }
        } else {
            audioExtractor.release()
            tempVideoFile.renameTo(outputFile)
        }

        VideoSwapResult(
            outputFile      = outputFile,
            statusMessage   = "Video swap complete. $framesProcessed frames processed.",
            framesProcessed = framesProcessed,
        )
    }
}
