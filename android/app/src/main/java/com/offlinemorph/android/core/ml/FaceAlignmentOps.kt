package com.offlinemorph.android.core.ml

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Rect
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.photo.Photo

/**
 * Canonical 5-point face templates (left eye, right eye, nose, left mouth, right mouth),
 * normalised to [0, 1] so they can be scaled to any crop size.
 */
enum class FaceTemplate(private val normalized: FloatArray) {
    /** InsightFace `arcface_dst` / 112 — the crop ArcFace recognisers were trained on. */
    ARCFACE_112(floatArrayOf(
        0.34191607f, 0.46157411f, 0.65653393f, 0.45983393f, 0.50022500f, 0.64050536f,
        0.37097589f, 0.82469196f, 0.63151696f, 0.82325089f,
    )),

    /**
     * InsightFace `estimate_norm(lmk, 128)` = `arcface_dst + (8, 0)` / 128 — the crop inswapper_128
     * was trained on. The +8 px x-offset is essential: without it the face sits off-centre.
     */
    ARCFACE_128(floatArrayOf(
        0.36167656f, 0.40387734f, 0.63696719f, 0.40235469f, 0.50019687f, 0.56044219f,
        0.38710391f, 0.72160547f, 0.61507734f, 0.72034453f,
    )),

    /** FFHQ alignment used by GFPGAN / CodeFormer style restorers. */
    FFHQ_512(floatArrayOf(
        0.37691676f, 0.46864664f, 0.62285697f, 0.46912813f, 0.50123859f, 0.61331904f,
        0.39308822f, 0.72541100f, 0.61150205f, 0.72490465f,
    ));

    fun points(size: Int): FloatArray = FloatArray(normalized.size) { normalized[it] * size }
}

object FaceAlignmentOps {
    const val ALIGNED_SIZE = 128

    /** Longest side at which the (smooth) Poisson colour correction is solved. */
    private const val POISSON_MAX_SIDE = 256

    /**
     * Least-squares similarity transform (rotation + uniform scale + translation) mapping the
     * 5 source [landmarks] onto [template] scaled to [size]. Equivalent to InsightFace's
     * `SimilarityTransform.estimate(lmk, dst)`.
     */
    fun alignmentMatrix(landmarks: FloatArray, template: FaceTemplate, size: Int): Matrix =
        similarityTransform(landmarks, template.points(size))

    fun similarityTransform(landmarks: FloatArray, targetPoints: FloatArray): Matrix {
        require(landmarks.size == 10 && targetPoints.size == 10) { "Expected 5 (x, y) landmark pairs" }

        var srcMeanX = 0f; var srcMeanY = 0f; var dstMeanX = 0f; var dstMeanY = 0f
        for (i in 0 until 5) {
            srcMeanX += landmarks[i * 2]; srcMeanY += landmarks[i * 2 + 1]
            dstMeanX += targetPoints[i * 2]; dstMeanY += targetPoints[i * 2 + 1]
        }
        srcMeanX /= 5f; srcMeanY /= 5f; dstMeanX /= 5f; dstMeanY /= 5f

        var srcVar = 0f
        var dot = 0f
        var cross = 0f
        for (i in 0 until 5) {
            val sx = landmarks[i * 2] - srcMeanX
            val sy = landmarks[i * 2 + 1] - srcMeanY
            val dx = targetPoints[i * 2] - dstMeanX
            val dy = targetPoints[i * 2 + 1] - dstMeanY
            srcVar += sx * sx + sy * sy
            dot += sx * dx + sy * dy
            cross += sx * dy - sy * dx
        }

        val eps = 1e-6f
        val scale = if (srcVar <= eps) 1f else hypot(dot, cross) / srcVar
        val norm = max(hypot(dot, cross), eps)
        val cos = if (dot == 0f && cross == 0f) 1f else dot / norm
        val sin = if (dot == 0f && cross == 0f) 0f else cross / norm

        val a = scale * cos
        val b = -scale * sin
        val c = scale * sin
        val d = scale * cos
        val tx = dstMeanX - (a * srcMeanX + b * srcMeanY)
        val ty = dstMeanY - (c * srcMeanX + d * srcMeanY)

        return Matrix().apply { setValues(floatArrayOf(a, b, tx, c, d, ty, 0f, 0f, 1f)) }
    }

    fun invert(matrix: Matrix): Matrix = Matrix().also { matrix.invert(it) }

    /**
     * Warps the region of [source] selected by [matrix] (source → crop space) into a
     * [size]×[size] RGB `CV_8UC3` crop. The caller owns and must release the returned [Mat].
     *
     * Only the source ROI that lands inside the crop is touched, and when the face is being
     * shrunk by more than ~1.5× it is first downsampled with INTER_AREA so the bilinear warp
     * does not alias (a 1000 px face squeezed into 128 px otherwise turns into noise).
     */
    fun warpToCrop(source: Bitmap, matrix: Matrix, size: Int): Mat {
        val roi = mappedBounds(invert(matrix), size.toFloat(), size.toFloat(), source.width, source.height, pad = 2)
            ?: return Mat.zeros(size, size, CvType.CV_8UC3)

        var input = bitmapRegionToRgbMat(source, roi)
        val local = Matrix(matrix).apply { preTranslate(roi.left.toFloat(), roi.top.toFloat()) }

        val scale = matrixScale(matrix)
        if (scale < 0.66f) {
            val w = max(1, (input.cols() * scale).roundToInt())
            val h = max(1, (input.rows() * scale).roundToInt())
            val sx = input.cols().toFloat() / w
            val sy = input.rows().toFloat() / h
            val resized = Mat()
            Imgproc.resize(input, resized, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
            input.release()
            input = resized
            // resize maps pixel centres: roi = (q + 0.5) * s - 0.5
            local.preTranslate(0.5f * (sx - 1f), 0.5f * (sy - 1f))
            local.preScale(sx, sy)
        }

        val affine = local.toAffineMat()
        val out = Mat()
        Imgproc.warpAffine(
            input, out, affine, Size(size.toDouble(), size.toDouble()),
            Imgproc.INTER_LINEAR, org.opencv.core.Core.BORDER_CONSTANT, Scalar(0.0, 0.0, 0.0),
        )
        input.release()
        affine.release()
        return out
    }

    /** Converts an RGB `CV_8UC3` [Mat] to a CHW RGB float tensor computed as `(pixel - mean) / std`. */
    fun rgbMatToChw(rgb: Mat, mean: Float, std: Float): FloatArray {
        val plane = rgb.rows() * rgb.cols()
        val bytes = ByteArray(plane * 3).also { rgb.get(0, 0, it) }
        val inv = 1f / std
        return FloatArray(plane * 3).also { out ->
            for (i in 0 until plane) {
                out[i] = ((bytes[i * 3].toInt() and 0xFF) - mean) * inv
                out[plane + i] = ((bytes[i * 3 + 1].toInt() and 0xFF) - mean) * inv
                out[plane * 2 + i] = ((bytes[i * 3 + 2].toInt() and 0xFF) - mean) * inv
            }
        }
    }

    /** Converts a CHW RGB float tensor in [0, 1] to an RGB `CV_8UC3` [Mat]. */
    fun chwToRgbMat(chw: FloatArray, size: Int): Mat {
        val plane = size * size
        require(chw.size == plane * 3) { "Expected CHW tensor of ${plane * 3} floats, got ${chw.size}" }
        val bytes = ByteArray(plane * 3)
        for (i in 0 until plane) {
            bytes[i * 3] = toByte(chw[i])
            bytes[i * 3 + 1] = toByte(chw[plane + i])
            bytes[i * 3 + 2] = toByte(chw[plane * 2 + i])
        }
        return Mat(size, size, CvType.CV_8UC3).apply { put(0, 0, bytes) }
    }

    private fun toByte(v: Float): Byte = (v * 255f + 0.5f).toInt().coerceIn(0, 255).toByte()

    /**
     * Pastes a square RGB [patch] back into [target] and returns a new bitmap.
     *
     * [patchToTarget] maps patch pixel coordinates into target coordinates. All work happens
     * inside the face's bounding ROI, so cost scales with face size rather than image size.
     *
     * The blend matte follows InsightFace's `paste_back`: warp a white square, erode by
     * mask_size/10 and Gaussian-blur by mask_size/20. When [seamless] is set, the face is
     * first Poisson-cloned (OpenCV NORMAL_CLONE) so its lighting and skin tone match the
     * target, and the soft matte is then applied on top so edges stay feathered.
     *
     * @param strength scales the matte, e.g. 0.8 keeps 20 % of the original pixels.
     */
    fun pasteBack(
        target: Bitmap,
        patch: Mat,
        patchToTarget: Matrix,
        seamless: Boolean,
        strength: Float = 1f,
    ): Bitmap {
        val patchSize = patch.cols().toFloat()
        val roi = mappedBounds(patchToTarget, patchSize, patch.rows().toFloat(), target.width, target.height, pad = 2)
            ?: return target
        val roiSize = Size(roi.width().toDouble(), roi.height().toDouble())

        val local = Matrix(patchToTarget).apply { postTranslate(-roi.left.toFloat(), -roi.top.toFloat()) }
        val affine = local.toAffineMat()
        val interpolation = if (matrixScale(patchToTarget) > 1f) Imgproc.INTER_CUBIC else Imgproc.INTER_LINEAR

        val targetRoi = bitmapRegionToRgbMat(target, roi)
        val warpedFace = Mat()
        Imgproc.warpAffine(patch, warpedFace, affine, roiSize, interpolation, org.opencv.core.Core.BORDER_REPLICATE)

        // Matte: warped white square → threshold → erode → blur.
        val white = Mat(patch.rows(), patch.cols(), CvType.CV_8UC1, Scalar(255.0))
        val mask = Mat()
        Imgproc.warpAffine(
            white, mask, affine, roiSize,
            Imgproc.INTER_LINEAR, org.opencv.core.Core.BORDER_CONSTANT, Scalar(0.0),
        )
        white.release()
        affine.release()
        Imgproc.threshold(mask, mask, 20.0, 255.0, Imgproc.THRESH_BINARY)
        val bounds = Imgproc.boundingRect(mask)
        val maskSize = sqrt((bounds.width * bounds.height).toDouble()).toInt()
        val erodeK = max(maskSize / 10, 10)
        val blurK = max(maskSize / 20, 5) * 2 + 1
        val kernel = Mat.ones(erodeK, erodeK, CvType.CV_8UC1)
        Imgproc.erode(mask, mask, kernel)
        kernel.release()
        Imgproc.GaussianBlur(mask, mask, Size(blurK.toDouble(), blurK.toDouble()), 0.0)

        var faceLayer = warpedFace
        if (seamless) {
            poissonClone(warpedFace, targetRoi, mask)?.let {
                warpedFace.release()
                faceLayer = it
            }
        }

        val matte = Mat()
        mask.convertTo(matte, CvType.CV_32F, strength / 255.0)
        val inverseMatte = Mat()
        matte.convertTo(inverseMatte, CvType.CV_32F, -1.0, 1.0)
        val blended = Mat()
        Imgproc.blendLinear(faceLayer, targetRoi, matte, inverseMatte, blended)

        val result = target.copy(Bitmap.Config.ARGB_8888, true)
        val roiBitmap = rgbMatToBitmap(blended)
        Canvas(result).drawBitmap(roiBitmap, roi.left.toFloat(), roi.top.toFloat(), null)
        roiBitmap.recycle()

        listOf(faceLayer, targetRoi, mask, matte, inverseMatte, blended).forEach { it.release() }
        return result
    }

    /**
     * Poisson-clones [src] into [dst] over every pixel the soft [matte] touches.
     *
     * NORMAL_CLONE's result is `src + h`, where `h` is harmonic with boundary values `dst − src`,
     * so `h` is smooth. It is therefore solved at ≤ [POISSON_MAX_SIDE] px and upsampled, which
     * gives the same colour correction at a fraction of the full-resolution solve cost.
     */
    private fun poissonClone(src: Mat, dst: Mat, matte: Mat): Mat? {
        val factor = min(1.0, POISSON_MAX_SIDE.toDouble() / max(src.cols(), src.rows()))
        if (factor >= 1.0) return seamlessCloneExact(src, dst, matte)

        val small = Size(max(3.0, src.cols() * factor).roundToInt().toDouble(), max(3.0, src.rows() * factor).roundToInt().toDouble())
        val srcSmall = Mat(); val dstSmall = Mat(); val matteSmall = Mat()
        Imgproc.resize(src, srcSmall, small, 0.0, 0.0, Imgproc.INTER_AREA)
        Imgproc.resize(dst, dstSmall, small, 0.0, 0.0, Imgproc.INTER_AREA)
        Imgproc.resize(matte, matteSmall, small, 0.0, 0.0, Imgproc.INTER_AREA)
        val clonedSmall = seamlessCloneExact(srcSmall, dstSmall, matteSmall)
        dstSmall.release(); matteSmall.release()
        if (clonedSmall == null) {
            srcSmall.release()
            return null
        }

        // correction = cloned − src at low resolution, upsampled and added back at full resolution.
        val correction = Mat()
        clonedSmall.convertTo(correction, CvType.CV_32FC3)
        val srcSmallF = Mat().also { srcSmall.convertTo(it, CvType.CV_32FC3) }
        org.opencv.core.Core.subtract(correction, srcSmallF, correction)
        val correctionFull = Mat()
        Imgproc.resize(correction, correctionFull, src.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
        val result = Mat().also { src.convertTo(it, CvType.CV_32FC3) }
        org.opencv.core.Core.add(result, correctionFull, result)
        result.convertTo(result, CvType.CV_8UC3)

        listOf(srcSmall, clonedSmall, correction, srcSmallF, correctionFull).forEach { it.release() }
        return result
    }

    /**
     * Runs OpenCV seamlessClone. The anchor is chosen so OpenCV's `roi_d = p − size / 2` lands
     * exactly on the mask's own bounding box (integer division), i.e. no one-pixel drift.
     */
    private fun seamlessCloneExact(src: Mat, dst: Mat, matte: Mat): Mat? {
        val cloneMask = Mat()
        return try {
            Imgproc.threshold(matte, cloneMask, 0.0, 255.0, Imgproc.THRESH_BINARY)
            // seamlessClone requires the mask to stay clear of the image border.
            Imgproc.rectangle(
                cloneMask, Point(0.0, 0.0),
                Point((cloneMask.cols() - 1).toDouble(), (cloneMask.rows() - 1).toDouble()),
                Scalar(0.0), 1,
            )
            val rect = Imgproc.boundingRect(cloneMask)
            if (rect.width < 3 || rect.height < 3) return null
            val anchor = Point((rect.x + rect.width / 2).toDouble(), (rect.y + rect.height / 2).toDouble())
            Mat().also { Photo.seamlessClone(src, dst, cloneMask, anchor, it, Photo.NORMAL_CLONE) }
        } catch (_: Exception) {
            null
        } finally {
            cloneMask.release()
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private fun matrixScale(matrix: Matrix): Float {
        val v = FloatArray(9).also(matrix::getValues)
        return sqrt(abs(v[Matrix.MSCALE_X] * v[Matrix.MSCALE_Y] - v[Matrix.MSKEW_X] * v[Matrix.MSKEW_Y]))
    }

    private fun Matrix.toAffineMat(): Mat {
        val v = FloatArray(9).also(::getValues)
        return Mat(2, 3, CvType.CV_64F).apply {
            put(
                0, 0,
                v[Matrix.MSCALE_X].toDouble(), v[Matrix.MSKEW_X].toDouble(), v[Matrix.MTRANS_X].toDouble(),
                v[Matrix.MSKEW_Y].toDouble(), v[Matrix.MSCALE_Y].toDouble(), v[Matrix.MTRANS_Y].toDouble(),
            )
        }
    }

    /** Bounding box of the [width]×[height] rectangle mapped through [matrix], clamped to the image. */
    private fun mappedBounds(
        matrix: Matrix,
        width: Float,
        height: Float,
        imageWidth: Int,
        imageHeight: Int,
        pad: Int,
    ): Rect? {
        val corners = floatArrayOf(0f, 0f, width, 0f, 0f, height, width, height)
        matrix.mapPoints(corners)
        var minX = Float.MAX_VALUE; var minY = Float.MAX_VALUE
        var maxX = -Float.MAX_VALUE; var maxY = -Float.MAX_VALUE
        for (i in 0 until 4) {
            minX = min(minX, corners[i * 2]); maxX = max(maxX, corners[i * 2])
            minY = min(minY, corners[i * 2 + 1]); maxY = max(maxY, corners[i * 2 + 1])
        }
        val rect = Rect(
            (floor(minX).toInt() - pad).coerceAtLeast(0),
            (floor(minY).toInt() - pad).coerceAtLeast(0),
            (ceil(maxX).toInt() + pad).coerceAtMost(imageWidth),
            (ceil(maxY).toInt() + pad).coerceAtMost(imageHeight),
        )
        return rect.takeIf { it.width() > 1 && it.height() > 1 }
    }

    private fun bitmapRegionToRgbMat(bitmap: Bitmap, roi: Rect): Mat {
        val software = if (bitmap.config == Bitmap.Config.ARGB_8888) bitmap
            else bitmap.copy(Bitmap.Config.ARGB_8888, false)
        val region = Bitmap.createBitmap(software, roi.left, roi.top, roi.width(), roi.height())
        val rgba = Mat()
        Utils.bitmapToMat(region, rgba)
        if (region !== bitmap) region.recycle()
        if (software !== bitmap && software !== region) software.recycle()
        val rgb = Mat()
        Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)
        rgba.release()
        return rgb
    }

    private fun rgbMatToBitmap(rgb: Mat): Bitmap {
        val rgba = Mat()
        Imgproc.cvtColor(rgb, rgba, Imgproc.COLOR_RGB2RGBA)
        val bitmap = Bitmap.createBitmap(rgb.cols(), rgb.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(rgba, bitmap)
        rgba.release()
        return bitmap
    }
}
