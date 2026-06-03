package com.offlinemorph.android.core.ml

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PointF
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.photo.Photo

object FaceAlignmentOps {
    const val ALIGNED_SIZE = 128

    // InsightFace-style 5-point anchors for 128×128, matching Python estimate_norm(lmk, 128).
    // Python logic: for size%112!=0 → ratio=size/128=1.0, diff_x=8.0*ratio=8.0
    //   dst = arcface_dst * ratio; dst[:,0] += diff_x
    // i.e. multiply y by 1.0 (unchanged), add 8 to every x.
    fun defaultTargetPoints128(): Array<PointF> {
        return arrayOf(
            PointF(38.2946f + 8.0f, 51.6963f),  // [46.2946, 51.6963]
            PointF(73.5318f + 8.0f, 51.5014f),  // [81.5318, 51.5014]
            PointF(56.0252f + 8.0f, 71.7366f),  // [64.0252, 71.7366]
            PointF(41.5493f + 8.0f, 92.3655f),  // [49.5493, 92.3655]
            PointF(70.7299f + 8.0f, 92.2041f),  // [78.7299, 92.2041]
        )
    }

    fun getSimilarityTransformMatrix(
        landmarks: FloatArray,
        targetPoints: Array<PointF>,
    ): Matrix {
        require(targetPoints.size == 5) { "targetPoints must contain 5 anchor points" }

        val srcFive = extractFivePointLandmarks(landmarks)
        val dstFive = targetPoints

        val srcMean = centroid(srcFive.asList())
        val dstMean = centroid(dstFive.asList())

        var srcVar = 0f
        var dot = 0f
        var cross = 0f

        for (i in 0 until 5) {
            val sx = srcFive[i].x - srcMean.x
            val sy = srcFive[i].y - srcMean.y
            val dx = dstFive[i].x - dstMean.x
            val dy = dstFive[i].y - dstMean.y

            srcVar += sx * sx + sy * sy
            dot += sx * dx + sy * dy
            cross += sx * dy - sy * dx
        }

        val eps = 1e-6f
        val scale = if (srcVar <= eps) 1f else hypot(dot, cross) / srcVar
        val cos = if (dot == 0f && cross == 0f) 1f else dot / max(hypot(dot, cross), eps)
        val sin = if (dot == 0f && cross == 0f) 0f else cross / max(hypot(dot, cross), eps)

        val a = scale * cos
        val b = -scale * sin
        val c = scale * sin
        val d = scale * cos

        val tx = dstMean.x - (a * srcMean.x + b * srcMean.y)
        val ty = dstMean.y - (c * srcMean.x + d * srcMean.y)

        val values = floatArrayOf(
            a, b, tx,
            c, d, ty,
            0f, 0f, 1f,
        )

        return Matrix().apply { setValues(values) }
    }

    fun preprocessTargetTensor(
        bitmap: Bitmap,
        affineMatrix: Matrix,
    ): FloatBuffer {
        val aligned = Bitmap.createBitmap(ALIGNED_SIZE, ALIGNED_SIZE, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(aligned)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
        canvas.drawBitmap(bitmap, affineMatrix, paint)

        val pixels = IntArray(ALIGNED_SIZE * ALIGNED_SIZE)
        aligned.getPixels(pixels, 0, ALIGNED_SIZE, 0, 0, ALIGNED_SIZE, ALIGNED_SIZE)
        aligned.recycle()

        val floats = ALIGNED_SIZE * ALIGNED_SIZE * 3
        val buffer = ByteBuffer
            .allocateDirect(floats * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()

        val plane = ALIGNED_SIZE * ALIGNED_SIZE
        var i = 0
        while (i < pixels.size) {
            val argb = pixels[i]
            val r = ((argb shr 16) and 0xFF) / 255.0f
            val g = ((argb shr 8) and 0xFF) / 255.0f
            val b = (argb and 0xFF) / 255.0f

            // CHW RGB — matches Python prepare_crop_frame: BGR->RGB, /255, HWC->CHW
            // Plane 0 = R, plane 1 = G, plane 2 = B
            buffer.put(i, r)
            buffer.put(plane + i, g)
            buffer.put(plane * 2 + i, b)
            i += 1
        }

        buffer.position(0)
        return buffer
    }

    /**
     * Composites a swapped face back into the original target image.
     *
     * Steps that match the Python paste_upscale path:
     * 1. Decode the CHW RGB tensor to a 128×128 bitmap.
     * 2. Bicubic-upscale to [upscaleToSize] (default 512) before warping, matching the
     *    Python scale_factor=4 pass that runs before paste_upscale().
     * 3. Compute a scaled inverse affine matrix so the 512×512 patch maps correctly
     *    back into the original image coordinate space.
     * 4. Warp the upscaled face and a solid-white mask to the target image space.
     * 5. Erode and Gaussian-blur the warped mask (matching Python blur_area()).
     * 6. Alpha-blend the result.
     */
    fun compositeSwapBack(
        swappedFaceTensor: FloatArray,
        originalTarget: Bitmap,
        inverseMatrix: Matrix,
        upscaleToSize: Int = 512,
    ): Bitmap {
        val plane = ALIGNED_SIZE * ALIGNED_SIZE
        require(swappedFaceTensor.size == plane * 3) {
            "Expected CHW tensor of size ${plane * 3}, got ${swappedFaceTensor.size}"
        }

        // 1. Decode CHW RGB tensor → 128×128 ARGB bitmap.
        //    Inswapper outputs RGB CHW (same order as its input); Python normalize_swap_frame
        //    does CHW→HWC * 255 → [::-1] to get BGR for OpenCV.  Plane 0 = R, 1 = G, 2 = B.
        val patchPixels = IntArray(plane)
        var i = 0
        while (i < plane) {
            val r = (swappedFaceTensor[i] * 255.0f).toInt().coerceIn(0, 255)
            val g = (swappedFaceTensor[plane + i] * 255.0f).toInt().coerceIn(0, 255)
            val b = (swappedFaceTensor[plane * 2 + i] * 255.0f).toInt().coerceIn(0, 255)
            patchPixels[i] = (255 shl 24) or (r shl 16) or (g shl 8) or b
            i += 1
        }
        val swappedPatch128 = Bitmap.createBitmap(patchPixels, ALIGNED_SIZE, ALIGNED_SIZE, Bitmap.Config.ARGB_8888)

        // 2. Lanczos4 upscale: 128 → upscaleToSize via OpenCV INTER_LANCZOS4.
        //    Hardware-accelerated C++ path; sharper than Android Bitmap bilinear scaling.
        val swappedPatch = lanczosUpscale(swappedPatch128, upscaleToSize)
        swappedPatch128.recycle()

        return compositeSwapBack(swappedPatch, originalTarget, inverseMatrix, upscaleToSize)
    }

    /**
     * Composites a pre-upscaled swapped face patch back into the original target image.
     *
     * Use this overload when [swappedPatch] has already been decoded and upscaled externally
     * (e.g. via [OnDeviceFaceEnhancer]). The [swappedPatch] bitmap is recycled on return.
     *
     * @param swappedPatch an [upscaleToSize]×[upscaleToSize] ARGB bitmap of the swapped face.
     * @param upscaleToSize must match the actual pixel dimensions of [swappedPatch].
     */
    fun compositeSwapBack(
        swappedPatch: Bitmap,
        originalTarget: Bitmap,
        inverseMatrix: Matrix,
        upscaleToSize: Int = 512,
    ): Bitmap {
        // Solid white mask at the upscaled size (matte before erosion/blur).
        val maskPatch = Bitmap.createBitmap(upscaleToSize, upscaleToSize, Bitmap.Config.ARGB_8888)
        Canvas(maskPatch).drawColor(Color.WHITE)

        // Scaled inverse matrix: maps upscaleToSize face space → target image space.
        val scaleFactor = ALIGNED_SIZE.toFloat() / upscaleToSize.toFloat()
        val scaledInverse = Matrix().apply {
            setScale(scaleFactor, scaleFactor)
            postConcat(inverseMatrix)
        }

        val width = originalTarget.width
        val height = originalTarget.height

        // Warp face and mask to target image space.
        val warpedFace = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val warpedMask = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val drawPaint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
        Canvas(warpedFace).drawBitmap(swappedPatch, scaledInverse, drawPaint)
        Canvas(warpedMask).drawBitmap(maskPatch, scaledInverse, drawPaint)

        // Build eroded+blurred matte at 1/4 resolution to avoid OOM on large images.
        val maskScale = 4
        val maskW = max(width / maskScale, 16)
        val maskH = max(height / maskScale, 16)

        val smallMaskBmp = Bitmap.createScaledBitmap(warpedMask, maskW, maskH, true)
        warpedMask.recycle()
        val smallMaskPx = IntArray(maskW * maskH)
        smallMaskBmp.getPixels(smallMaskPx, 0, maskW, 0, 0, maskW, maskH)
        smallMaskBmp.recycle()

        val binaryMask = IntArray(maskW * maskH) { i -> (smallMaskPx[i] ushr 24) and 0xFF }
        for (x in 0 until maskW) { binaryMask[x] = 0; binaryMask[(maskH - 1) * maskW + x] = 0 }
        for (y in 0 until maskH) { binaryMask[y * maskW] = 0; binaryMask[y * maskW + maskW - 1] = 0 }

        val maskSize = estimateMaskSize(binaryMask, maskW, maskH)
        val blurAmount = 20
        val kErode = max(maskSize / (blurAmount / 2), blurAmount / 2)
        val kBlur   = max(maskSize / blurAmount, blurAmount / 5)

        val eroded   = erodeBinaryMask(binaryMask, maskW, maskH, kernelHalf = kErode / 2)
        val erodedF  = FloatArray(maskW * maskH) { i -> eroded[i] / 255f }
        val blurredF = separableBoxBlur(erodedF, maskW, maskH, radius = kBlur)

        val matteLowPx = IntArray(maskW * maskH) { i ->
            val a = (blurredF[i] * 255f).toInt().coerceIn(0, 255)
            (a shl 24) or 0x00FFFFFF
        }
        val matteLow = Bitmap.createBitmap(maskW, maskH, Bitmap.Config.ARGB_8888)
        matteLow.setPixels(matteLowPx, 0, maskW, 0, 0, maskW, maskH)

        val matteScaleMatrix = Matrix().apply {
            setScale(width.toFloat() / maskW, height.toFloat() / maskH)
        }
        Canvas(warpedFace).drawBitmap(
            matteLow, matteScaleMatrix,
            Paint().apply { xfermode = PorterDuffXfermode(PorterDuff.Mode.DST_IN) },
        )
        matteLow.recycle()

        val out = originalTarget.copy(Bitmap.Config.ARGB_8888, true)
        Canvas(out).drawBitmap(warpedFace, 0f, 0f, null)

        // Poisson-blend pass: replace the alpha-composited seam with seamlessClone for
        // photorealistic colour and gradient continuity at the face boundary.
        val blended = runCatching {
            poissonSeamBlend(src = warpedFace, dst = out, mask = warpedFace)
        }.getOrNull()

        swappedPatch.recycle()
        maskPatch.recycle()
        warpedFace.recycle()

        return blended ?: out
    }

    // ── Matte helpers ─────────────────────────────────────────────────────────

    /**
     * Estimates face region size from the bounds of non-zero pixels in the warped mask.
     * Mirrors Python: mask_size = int(sqrt(mask_h * mask_w)).
     */
    private fun estimateMaskSize(mask: IntArray, width: Int, height: Int): Int {
        var minH = height; var maxH = 0; var minW = width; var maxW = 0
        for (y in 0 until height) {
            for (x in 0 until width) {
                if (mask[y * width + x] > 128) {
                    if (y < minH) minH = y
                    if (y > maxH) maxH = y
                    if (x < minW) minW = x
                    if (x > maxW) maxW = x
                }
            }
        }
        if (maxH < minH || maxW < minW) return 64
        val maskH = (maxH - minH).coerceAtLeast(1)
        val maskW = (maxW - minW).coerceAtLeast(1)
        return sqrt((maskH * maskW).toFloat()).toInt().coerceAtLeast(10)
    }

    /**
     * Separable morphological erosion (rectangular structuring element).
     * Equivalent to cv2.erode with a square kernel.
     * O(W×H×k) via two sliding-minimum passes.
     */
    private fun erodeBinaryMask(mask: IntArray, width: Int, height: Int, kernelHalf: Int): IntArray {
        // Horizontal pass
        val temp = IntArray(width * height)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val xStart = (x - kernelHalf).coerceAtLeast(0)
                val xEnd = (x + kernelHalf).coerceAtMost(width - 1)
                var minVal = 255
                for (kx in xStart..xEnd) {
                    val v = mask[y * width + kx]
                    if (v < minVal) minVal = v
                }
                temp[y * width + x] = minVal
            }
        }
        // Vertical pass
        val output = IntArray(width * height)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val yStart = (y - kernelHalf).coerceAtLeast(0)
                val yEnd = (y + kernelHalf).coerceAtMost(height - 1)
                var minVal = 255
                for (ky in yStart..yEnd) {
                    val v = temp[ky * width + x]
                    if (v < minVal) minVal = v
                }
                output[y * width + x] = minVal
            }
        }
        return output
    }

    /**
     * Separable box blur used to soften the eroded matte.
     * Equivalent to cv2.GaussianBlur (box blur is a good approximation for mask smoothing).
     */
    private fun separableBoxBlur(input: FloatArray, width: Int, height: Int, radius: Int): FloatArray {
        val r = radius.coerceAtLeast(1)
        // Horizontal pass
        val temp = FloatArray(width * height)
        for (y in 0 until height) {
            for (x in 0 until width) {
                var sum = 0f
                var count = 0
                for (dx in -r..r) {
                    val nx = (x + dx).coerceIn(0, width - 1)
                    sum += input[y * width + nx]
                    count++
                }
                temp[y * width + x] = sum / count
            }
        }
        // Vertical pass
        val output = FloatArray(width * height)
        for (y in 0 until height) {
            for (x in 0 until width) {
                var sum = 0f
                var count = 0
                for (dy in -r..r) {
                    val ny = (y + dy).coerceIn(0, height - 1)
                    sum += temp[ny * width + x]
                    count++
                }
                output[y * width + x] = sum / count
            }
        }
        return output
    }

    private fun extractFivePointLandmarks(landmarks: FloatArray): Array<PointF> {
        if (landmarks.size == 10) {
            return Array(5) { i -> PointF(landmarks[i * 2], landmarks[i * 2 + 1]) }
        }

        require(landmarks.size >= 20 && landmarks.size % 2 == 0) {
            "landmarks must be either 5-point (10 floats) or dense even-length array"
        }

        val points = Array(landmarks.size / 2) { idx ->
            PointF(landmarks[idx * 2], landmarks[idx * 2 + 1])
        }

        var minX = Float.POSITIVE_INFINITY
        var minY = Float.POSITIVE_INFINITY
        var maxX = Float.NEGATIVE_INFINITY
        var maxY = Float.NEGATIVE_INFINITY
        for (p in points) {
            if (p.x < minX) minX = p.x
            if (p.y < minY) minY = p.y
            if (p.x > maxX) maxX = p.x
            if (p.y > maxY) maxY = p.y
        }

        val width = max(maxX - minX, 1f)
        val height = max(maxY - minY, 1f)
        val midX = minX + width * 0.5f

        val eyeBandTop = minY + height * 0.18f
        val eyeBandBottom = minY + height * 0.62f
        val mouthBandTop = minY + height * 0.60f

        val leftEyeCandidates = points.filter { it.x < midX && it.y in eyeBandTop..eyeBandBottom }
        val rightEyeCandidates = points.filter { it.x >= midX && it.y in eyeBandTop..eyeBandBottom }
        val noseCandidates = points.filter {
            it.x in (minX + width * 0.35f)..(minX + width * 0.65f) &&
                it.y in (minY + height * 0.35f)..(minY + height * 0.75f)
        }
        val mouthCandidates = points.filter { it.y >= mouthBandTop }

        val leftEye = if (leftEyeCandidates.isNotEmpty()) centroid(leftEyeCandidates) else PointF(minX + width * 0.32f, minY + height * 0.42f)
        val rightEye = if (rightEyeCandidates.isNotEmpty()) centroid(rightEyeCandidates) else PointF(minX + width * 0.68f, minY + height * 0.42f)
        val nose = if (noseCandidates.isNotEmpty()) centroid(noseCandidates) else PointF(minX + width * 0.5f, minY + height * 0.60f)

        val mouthLeft = mouthCandidates.minByOrNull { it.x } ?: PointF(minX + width * 0.38f, minY + height * 0.80f)
        val mouthRight = mouthCandidates.maxByOrNull { it.x } ?: PointF(minX + width * 0.62f, minY + height * 0.80f)

        return arrayOf(leftEye, rightEye, nose, mouthLeft, mouthRight)
    }

    /**
     * Upscales [src] to [targetSize]×[targetSize] using OpenCV INTER_LANCZOS4.
     *
     * INTER_LANCZOS4 uses an 8×8 Lanczos kernel and preserves high-frequency detail
     * far better than Android's built-in bilinear scaler, which matters visibly when
     * mapping a 128×128 inswapper patch back to a 1080p target canvas.
     *
     * All native [Mat] allocations are released explicitly to avoid C++ heap leaks.
     */
    private fun lanczosUpscale(src: Bitmap, targetSize: Int): Bitmap {
        val srcMat = Mat()
        Utils.bitmapToMat(src, srcMat)
        val dstMat = Mat()
        Imgproc.resize(
            srcMat,
            dstMat,
            Size(targetSize.toDouble(), targetSize.toDouble()),
            0.0,
            0.0,
            Imgproc.INTER_LANCZOS4,
        )
        srcMat.release()
        val result = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(dstMat, result)
        dstMat.release()
        return result
    }

    /**
     * Applies OpenCV seamlessClone (NORMAL_CLONE) to smooth the seam between [src] and [dst].
     *
     * [src] is the warped swapped-face bitmap (ARGB, alpha used to derive the blend mask).
     * [dst] is the current composited output to refine.
     *
     * Poisson blending preserves the face's gradient field while matching the target's
     * colour and luminance at the boundary, eliminating the halo effect visible with
     * simple alpha compositing.
     *
     * Returns a new bitmap on success; returns null on any OpenCV error so the caller
     * can fall back to the plain alpha-blended result.
     */
    private fun poissonSeamBlend(src: Bitmap, dst: Bitmap, mask: Bitmap): Bitmap? {
        val srcMat = Mat()
        val dstMat = Mat()
        val maskMat = Mat()
        val resultMat = Mat()
        return try {
            Utils.bitmapToMat(src, srcMat)
            Utils.bitmapToMat(dst, dstMat)
            Utils.bitmapToMat(mask, maskMat)

            // Build an 8-bit grayscale matte from the mask alpha channel.
            val channels = java.util.ArrayList<Mat>()
            org.opencv.core.Core.split(maskMat, channels)
            val alphaCh = channels[3]
            val grayMask = Mat()
            // Threshold: any pixel with alpha > 10 contributes to the blend region.
            Imgproc.threshold(alphaCh, grayMask, 10.0, 255.0, Imgproc.THRESH_BINARY)
            alphaCh.release()
            channels.forEach { it.release() }

            // seamlessClone requires BGR (3-channel) sources.
            val srcBgr = Mat()
            val dstBgr = Mat()
            Imgproc.cvtColor(srcMat, srcBgr, Imgproc.COLOR_RGBA2BGR)
            Imgproc.cvtColor(dstMat, dstBgr, Imgproc.COLOR_RGBA2BGR)

            // Centre point of the non-zero mask region used as the clone anchor.
            val moments = Imgproc.moments(grayMask)
            val cx = if (moments.m00 > 0) (moments.m10 / moments.m00) else (dst.width / 2.0)
            val cy = if (moments.m00 > 0) (moments.m01 / moments.m00) else (dst.height / 2.0)
            val centre = Point(cx, cy)

            val blendedBgr = Mat()
            Photo.seamlessClone(srcBgr, dstBgr, grayMask, centre, blendedBgr, Photo.NORMAL_CLONE)

            val result = Bitmap.createBitmap(dst.width, dst.height, Bitmap.Config.ARGB_8888)
            Imgproc.cvtColor(blendedBgr, resultMat, Imgproc.COLOR_BGR2RGBA)
            Utils.matToBitmap(resultMat, result)

            srcBgr.release(); dstBgr.release(); blendedBgr.release(); grayMask.release()
            result
        } catch (_: Exception) {
            null
        } finally {
            srcMat.release(); dstMat.release(); maskMat.release(); resultMat.release()
        }
    }

    private fun centroid(points: List<PointF>): PointF {
        var sx = 0f
        var sy = 0f
        for (p in points) {
            sx += p.x
            sy += p.y
        }
        val n = max(points.size, 1).toFloat()
        return PointF(sx / n, sy / n)
    }
}
