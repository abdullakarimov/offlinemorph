package com.offlinemorph.android.core.ml.face

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Rect
import com.offlinemorph.android.core.ml.FaceAlignmentOps
import com.offlinemorph.android.core.ml.FaceTemplate
import com.offlinemorph.android.core.ml.OrtSessionFactory
import com.offlinemorph.android.core.ml.OrtValueUtils
import com.offlinemorph.android.feature.models.ModelCatalog
import java.io.File
import kotlin.math.hypot
import kotlin.math.max
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/** Region ids of the 19-class BiSeNet face-parsing model (CelebAMask-HQ label order). */
object FaceClass {
    const val BACKGROUND = 0
    const val SKIN = 1
    const val L_BROW = 2
    const val R_BROW = 3
    const val L_EYE = 4
    const val R_EYE = 5
    const val GLASSES = 6
    const val L_EAR = 7
    const val R_EAR = 8
    const val EARRING = 9
    const val NOSE = 10
    const val MOUTH = 11
    const val U_LIP = 12
    const val L_LIP = 13
    const val NECK = 14
    const val NECKLACE = 15
    const val CLOTH = 16
    const val HAIR = 17
    const val HAT = 18
}

/**
 * Semantic regions of one face, held in an aligned [FaceRegionParser.SIZE]² crop.
 *
 * Masks are built in crop space, where the face is upright and has a fixed scale, and then
 * [project]ed onto the full-resolution image, so edits are applied at the photo's own
 * resolution rather than at the crop's.
 *
 * @param classMap   per-pixel [FaceClass] id, row-major, SIZE × SIZE.
 * @param imageToCrop maps image pixels to crop pixels.
 * @param fromModel  true when produced by the parsing model, false for the landmark template fallback.
 * @param zoom       template shrink factor used for this crop (smaller = wider view).
 */
class FaceParse(
    val classMap: ByteArray,
    val imageToCrop: Matrix,
    val keypointsInCrop: FloatArray,
    val fromModel: Boolean,
    /** Template shrink factor used for this crop; see [FaceRegionParser.ZOOM]. */
    val zoom: Float,
    imageWidth: Int,
    imageHeight: Int,
) {
    val size: Int = FaceRegionParser.SIZE
    val cropToImage: Matrix = FaceAlignmentOps.invert(imageToCrop)

    /** Image-space bounds of the crop footprint, clamped to the image. Null when off-image. */
    val roi: Rect? = FaceAlignmentOps.mappedBounds(
        cropToImage, size.toFloat(), size.toFloat(), imageWidth, imageHeight, pad = 2,
    )

    /** Inter-ocular distance in crop pixels. */
    val eyeDistance: Float get() = hypot(
        keypointsInCrop[2] - keypointsInCrop[0], keypointsInCrop[3] - keypointsInCrop[1],
    )

    /** Image pixels per crop pixel. */
    val imageScale: Float get() = 1f / FaceAlignmentOps.matrixScale(imageToCrop)

    /** Binary CV_8UC1 mask (0/255) of the given classes. Caller releases. */
    fun classMask(vararg classes: Int): Mat {
        val lookup = BooleanArray(256).also { t -> classes.forEach { t[it] = true } }
        val bytes = ByteArray(classMap.size) { i -> if (lookup[classMap[i].toInt() and 0xFF]) -1 else 0 }
        return Mat(size, size, CvType.CV_8UC1).apply { put(0, 0, bytes) }
    }

    /**
     * Point given in normalised FFHQ-template coordinates (eyes ≈ (0.38, 0.47) / (0.62, 0.47))
     * mapped into this crop.
     */
    fun templatePoint(nx: Float, ny: Float): Point = Point(
        ((nx - 0.5f) * zoom + 0.5f) * size.toDouble(),
        ((ny - 0.5f) * zoom + 0.5f) * size.toDouble(),
    )

    /**
     * Feathers a crop-space mask (CV_8UC1 0–255 or CV_32FC1 0–1) by [featherPx] crop pixels and
     * warps it onto [roi]. Returns a CV_32FC1 mask in [0, 1] of the ROI's size, or null when
     * the face is off-image. Caller releases.
     */
    fun project(cropMask: Mat, featherPx: Int): Mat? = roi?.let { projectInto(cropMask, it, featherPx) }

    /** Like [project], but onto an arbitrary image-space [target] rectangle. Caller releases. */
    fun projectInto(cropMask: Mat, target: Rect, featherPx: Int): Mat {
        val roi = target
        val soft = Mat()
        if (cropMask.type() == CvType.CV_32FC1) cropMask.copyTo(soft) else cropMask.convertTo(soft, CvType.CV_32F, 1.0 / 255.0)
        if (featherPx > 0) {
            val k = featherPx * 2 + 1
            Imgproc.GaussianBlur(soft, soft, Size(k.toDouble(), k.toDouble()), 0.0)
        }
        val local = Matrix(cropToImage).apply { postTranslate(-roi.left.toFloat(), -roi.top.toFloat()) }
        val affine = local.toAffine()
        val out = Mat()
        Imgproc.warpAffine(
            soft, out, affine, Size(roi.width().toDouble(), roi.height().toDouble()),
            Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, Scalar(0.0),
        )
        soft.release()
        affine.release()
        return out
    }

    /**
     * Affine coefficients (a, b, c, d, e, f) mapping ROI pixel (x, y) to crop space as
     * (a·x + b·y + c, d·x + e·y + f). Used for effects that follow the face's own axes
     * (ombre, highlights) regardless of head tilt.
     */
    fun roiToCropAffine(): FloatArray {
        val roi = roi ?: return floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f)
        val v = FloatArray(9).also(imageToCrop::getValues)
        val a = v[Matrix.MSCALE_X]; val b = v[Matrix.MSKEW_X]; val c = v[Matrix.MTRANS_X]
        val d = v[Matrix.MSKEW_Y]; val e = v[Matrix.MSCALE_Y]; val f = v[Matrix.MTRANS_Y]
        return floatArrayOf(a, b, c + a * roi.left + b * roi.top, d, e, f + d * roi.left + e * roi.top)
    }
}

/**
 * Produces [FaceParse]s: the BiSeNet face-parsing model ([ModelCatalog.FACE_PARSING]) when it is
 * installed, otherwise an approximate parse drawn from the 5-point landmarks (skin oval, eyes,
 * brows, nose, lips, inner mouth), so landmark-driven edits still work without the model.
 */
class FaceRegionParser(
    private val modelsDirectory: File,
    private val sessionFactory: OrtSessionFactory = OrtSessionFactory(),
) {
    companion object {
        const val SIZE = 512

        /**
         * Shrinks the FFHQ template toward the crop centre so a little more hair and jaw fit in
         * the crop than in the tight FFHQ framing the parser was trained near.
         */
        const val ZOOM = 0.9f

        /** Wider framing for hair, which extends well beyond the face. */
        const val HAIR_ZOOM = 0.6f

        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    fun hasModel(): Boolean = File(modelsDirectory, ModelCatalog.FACE_PARSING).isFile

    /** Similarity transform from image pixels to the parser crop for the face at [keypoints]. */
    fun alignment(keypoints: FloatArray, zoom: Float = ZOOM): Matrix {
        val template = FaceTemplate.FFHQ_512.points(1)
        val target = FloatArray(10) { ((template[it] - 0.5f) * zoom + 0.5f) * SIZE }
        return FaceAlignmentOps.similarityTransform(keypoints, target)
    }

    /**
     * Parses the face at [keypoints] in [image]. Falls back to the template parse when
     * [useModel] is false or the model is missing. A smaller [zoom] frames more of the head
     * (e.g. [HAIR_ZOOM] so long hair fits), at lower resolution per face pixel.
     */
    fun parse(image: Bitmap, keypoints: FloatArray, useModel: Boolean = true, zoom: Float = ZOOM): FaceParse {
        val toCrop = alignment(keypoints, zoom)
        val cropKps = keypoints.copyOf().also { toCrop.mapPoints(it) }
        val classMap = if (useModel && hasModel()) {
            runModel(image, toCrop)
        } else {
            templateParse(cropKps)
        }
        return FaceParse(classMap, toCrop, cropKps, useModel && hasModel(), zoom, image.width, image.height)
    }

    private fun runModel(image: Bitmap, toCrop: Matrix): ByteArray {
        val crop = FaceAlignmentOps.warpToCrop(image, toCrop, SIZE)
        val plane = SIZE * SIZE
        val bytes = ByteArray(plane * 3).also { crop.get(0, 0, it) }
        crop.release()
        val input = FloatArray(plane * 3)
        for (i in 0 until plane) {
            for (c in 0 until 3) {
                input[c * plane + i] = ((bytes[i * 3 + c].toInt() and 0xFF) / 255f - MEAN[c]) / STD[c]
            }
        }
        val session = sessionFactory.session(File(modelsDirectory, ModelCatalog.FACE_PARSING))
        val logits = sessionFactory.createFloatTensor(input, longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong())).use { t ->
            session.run(mapOf(session.inputNames.first() to t)).use { result ->
                OrtValueUtils.extractFloatOutputs(result).firstOrNull()?.data
                    ?: error("Face parsing model returned no output")
            }
        }
        val classes = logits.size / plane
        require(classes >= 19) { "Unexpected face parsing output of ${logits.size} values" }
        return ByteArray(plane) { i ->
            var best = 0
            var bestValue = logits[i]
            for (c in 1 until classes) {
                val v = logits[c * plane + i]
                if (v > bestValue) { bestValue = v; best = c }
            }
            best.toByte()
        }
    }

    /** Approximate parse drawn from crop-space 5-point keypoints. */
    private fun templateParse(k: FloatArray): ByteArray {
        val mat = Mat.zeros(SIZE, SIZE, CvType.CV_8UC1)
        val eyeL = Point(k[0].toDouble(), k[1].toDouble())
        val eyeR = Point(k[2].toDouble(), k[3].toDouble())
        val nose = Point(k[4].toDouble(), k[5].toDouble())
        val mouthL = Point(k[6].toDouble(), k[7].toDouble())
        val mouthR = Point(k[8].toDouble(), k[9].toDouble())
        val iod = hypot(eyeR.x - eyeL.x, eyeR.y - eyeL.y)
        val mouthW = max(hypot(mouthR.x - mouthL.x, mouthR.y - mouthL.y), iod * 0.5)
        val mouthC = Point((mouthL.x + mouthR.x) / 2, (mouthL.y + mouthR.y) / 2)
        val eyeC = Point((eyeL.x + eyeR.x) / 2, (eyeL.y + eyeR.y) / 2)

        fun fill(center: Point, ax: Double, ay: Double, cls: Int) =
            Imgproc.ellipse(mat, center, Size(ax, ay), 0.0, 0.0, 360.0, Scalar(cls.toDouble()), -1)

        // Face oval: from above the brows to below the chin.
        val faceCenter = Point(eyeC.x, eyeC.y + (mouthC.y - eyeC.y) * 0.55)
        fill(faceCenter, iod * 1.05, (mouthC.y - eyeC.y) * 1.85, FaceClass.SKIN)
        fill(Point(nose.x, nose.y - iod * 0.15), iod * 0.22, iod * 0.35, FaceClass.NOSE)
        fill(Point(eyeL.x, eyeL.y - iod * 0.33), iod * 0.30, iod * 0.07, FaceClass.L_BROW)
        fill(Point(eyeR.x, eyeR.y - iod * 0.33), iod * 0.30, iod * 0.07, FaceClass.R_BROW)
        fill(eyeL, iod * 0.24, iod * 0.10, FaceClass.L_EYE)
        fill(eyeR, iod * 0.24, iod * 0.10, FaceClass.R_EYE)
        fill(Point(mouthC.x, mouthC.y - mouthW * 0.08), mouthW * 0.58, mouthW * 0.16, FaceClass.U_LIP)
        fill(Point(mouthC.x, mouthC.y + mouthW * 0.12), mouthW * 0.55, mouthW * 0.18, FaceClass.L_LIP)
        fill(mouthC, mouthW * 0.42, mouthW * 0.07, FaceClass.MOUTH)

        val out = ByteArray(SIZE * SIZE).also { mat.get(0, 0, it) }
        mat.release()
        return out
    }
}

/** Reads and writes the [FaceParse.roi] of an image as an RGB `CV_8UC3` [Mat]. */
object RoiIo {
    fun read(bitmap: Bitmap, roi: Rect): Mat = FaceAlignmentOps.bitmapRegionToRgbMat(bitmap, roi)

    /** Draws [rgb] at [roi] into [target], which must be mutable. */
    fun write(target: Bitmap, roi: Rect, rgb: Mat) {
        val patch = FaceAlignmentOps.rgbMatToBitmap(rgb)
        Canvas(target).drawBitmap(patch, roi.left.toFloat(), roi.top.toFloat(), null)
        patch.recycle()
    }
}

internal fun Matrix.toAffine(): Mat {
    val v = FloatArray(9).also(::getValues)
    return Mat(2, 3, CvType.CV_64F).apply {
        put(
            0, 0,
            v[Matrix.MSCALE_X].toDouble(), v[Matrix.MSKEW_X].toDouble(), v[Matrix.MTRANS_X].toDouble(),
            v[Matrix.MSKEW_Y].toDouble(), v[Matrix.MSCALE_Y].toDouble(), v[Matrix.MTRANS_Y].toDouble(),
        )
    }
}

/** Translates a CV_8UC1 mask by ([dx], [dy]) pixels, filling with 0. Caller releases. */
internal fun Mat.shifted(dx: Double, dy: Double): Mat {
    val m = Mat(2, 3, CvType.CV_64F).apply { put(0, 0, 1.0, 0.0, dx, 0.0, 1.0, dy) }
    val out = Mat()
    Imgproc.warpAffine(this, out, m, size(), Imgproc.INTER_NEAREST, Core.BORDER_CONSTANT, Scalar(0.0))
    m.release()
    return out
}

/** Morphological dilation with an elliptical kernel of diameter [k]. Caller releases. */
internal fun Mat.dilated(k: Int): Mat {
    val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(k.toDouble(), k.toDouble()))
    val out = Mat()
    Imgproc.dilate(this, out, kernel)
    kernel.release()
    return out
}

/** Morphological erosion with an elliptical kernel of diameter [k]. Caller releases. */
internal fun Mat.eroded(k: Int): Mat {
    val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(k.toDouble(), k.toDouble()))
    val out = Mat()
    Imgproc.erode(this, out, kernel)
    kernel.release()
    return out
}
