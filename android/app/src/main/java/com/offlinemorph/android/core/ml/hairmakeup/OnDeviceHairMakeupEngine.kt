package com.offlinemorph.android.core.ml.hairmakeup

import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import com.offlinemorph.android.core.ml.EngineError
import com.offlinemorph.android.core.ml.EngineProgress
import com.offlinemorph.android.core.ml.EngineResult
import com.offlinemorph.android.core.ml.FaceAnalyzer
import com.offlinemorph.android.core.ml.OrtSessionFactory
import com.offlinemorph.android.core.ml.face.ColorLab
import com.offlinemorph.android.core.ml.face.FaceClass
import com.offlinemorph.android.core.ml.face.FaceParse
import com.offlinemorph.android.core.ml.face.FaceRegionParser
import com.offlinemorph.android.core.ml.face.RegionEffects
import com.offlinemorph.android.core.ml.face.RegionEffects.Blend
import com.offlinemorph.android.core.ml.face.RoiIo
import com.offlinemorph.android.core.ml.face.dilated
import com.offlinemorph.android.core.ml.face.faceKeypoints
import com.offlinemorph.android.core.ml.face.shifted
import com.offlinemorph.android.feature.models.ModelCatalog
import java.io.File
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.random.Random
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * On-device hair & makeup studio backed by the BiSeNet face-parsing model.
 *
 * Every detected face (up to [MAX_FACES]) is aligned and parsed in its own crop; masks are
 * built in that upright, scale-normalised crop and projected back onto the photo at full
 * resolution:
 *
 * - **Hair** (wider [FaceRegionParser.HAIR_ZOOM] crop so long hair fits): Lab recolour that keeps
 *   the hair's own shading, with solid, ombré (root → tip gradient along the face's vertical
 *   axis) or highlight-streak layouts.
 * - **Lips**: Lab recolour that keeps gloss highlights.
 * - **Eyeshadow**: lid band above each eye, below the brow, soft-light blended.
 * - **Eyeliner**: thin band along the upper eye contour.
 * - **Blush**: soft cheek blobs clipped to skin.
 * - **Facial hair**: procedural strand texture inside a style-specific region (moustache,
 *   goatee, stubble or full beard) bounded by the parsed skin, so it follows the jaw line.
 */
class OnDeviceHairMakeupEngine(
    private val modelsDirectory: File,
    private val faceAnalyzer: FaceAnalyzer,
    sessionFactory: OrtSessionFactory = OrtSessionFactory(),
) : HairMakeupEngine {

    private val parser = FaceRegionParser(modelsDirectory, sessionFactory)

    override suspend fun runHairMakeup(
        request: HairMakeupRequest,
        onProgress: (EngineProgress) -> Unit,
    ): EngineResult<HairMakeupResult> {
        for (model in listOf(ModelCatalog.FACE_PARSING, ModelCatalog.DETECTOR)) {
            if (!File(modelsDirectory, model).isFile) {
                return EngineResult.Failure(
                    error = EngineError.ModelNotFound(model),
                    statusMessage = "$model is not installed. Open Setup → Download All to get it.",
                )
            }
        }
        if (!request.hasAnyEffect) {
            return EngineResult.Success(HairMakeupResult(request.sourceBitmap.copy(Bitmap.Config.ARGB_8888, false)))
        }
        val intensity = request.intensity.coerceIn(0f, 1f)

        return try {
            onProgress(EngineProgress("Detecting faces…", 0.05f))
            val faces = faceAnalyzer.analyze(request.sourceBitmap).faceKeypoints(MAX_FACES)
            if (faces.isEmpty()) {
                return EngineResult.Failure(EngineError.NoFaceDetected("hair & makeup"), "No face found in the photo.")
            }

            val out = request.sourceBitmap.copy(Bitmap.Config.ARGB_8888, true)
            for ((index, kps) in faces.withIndex()) {
                val label = if (faces.size > 1) " (face ${index + 1} / ${faces.size})" else ""
                if (request.hairColorHex != null) {
                    onProgress(EngineProgress("Segmenting hair$label…", 0.1f + 0.85f * index / faces.size))
                    applyHair(out, parser.parse(out, kps, zoom = FaceRegionParser.HAIR_ZOOM), request, intensity)
                }
                onProgress(EngineProgress("Applying makeup$label…", 0.1f + 0.85f * (index + 0.5f) / faces.size))
                applyFace(out, parser.parse(out, kps), request, intensity)
            }
            onProgress(EngineProgress("Done", 1f))
            EngineResult.Success(HairMakeupResult(out))
        } catch (e: Exception) {
            Log.e(TAG, "Hair & makeup failed", e)
            EngineResult.Failure(EngineError.InferenceFailure(e), e.message ?: "Hair & makeup processing failed.")
        }
    }

    // ── Hair ─────────────────────────────────────────────────────────────────

    private fun applyHair(image: Bitmap, parse: FaceParse, request: HairMakeupRequest, intensity: Float) {
        val roi = parse.roi ?: return
        val base = parseColor(request.hairColorHex ?: return)
        val secondary = request.hairSecondaryColorHex?.let(::parseColor) ?: lighter(base, 28f)

        val hairCrop = parse.classMask(FaceClass.HAIR)
        val featherPx = max(2, (parse.eyeDistance * 0.03f).roundToInt())
        val mask = parse.project(hairCrop, featherPx) ?: run { hairCrop.release(); return }
        val hairRows = rowExtent(hairCrop)
        hairCrop.release()

        val rgb = RoiIo.read(image, roi)
        val cols = rgb.cols()
        val affine = parse.roiToCropAffine()
        val targetAt: (Int) -> Int = when (request.hairStyle) {
            HairStyle.SOLID -> { _ -> base }
            HairStyle.OMBRE -> { i ->
                val x = i % cols; val y = i / cols
                val cy = affine[3] * x + affine[4] * y + affine[5]
                val t = ((cy - hairRows.first) / max(1f, hairRows.second - hairRows.first)).coerceIn(0f, 1f)
                // Hold the root colour for the top third, then blend smoothly to the tips.
                lerpRgb(base, secondary, smooth(((t - 0.3f) / 0.6f).coerceIn(0f, 1f)))
            }
            HairStyle.HIGHLIGHTS -> { i ->
                val x = i % cols; val y = i / cols
                val cx = affine[0] * x + affine[1] * y + affine[2]
                val cy = affine[3] * x + affine[4] * y + affine[5]
                // 1-D value noise across the face's horizontal axis, gently waving down the hair.
                val s = valueNoise(cx / 7f + 0.6f * sin(cy / 40f))
                lerpRgb(base, secondary, smooth(((s - 0.58f) / 0.17f).coerceIn(0f, 1f)))
            }
        }
        RegionEffects.recolor(
            rgb, mask, strength = intensity * 0.9f, lightnessShift = 0.55f, chromaKeep = 0.2f,
            targetAt = targetAt,
        )
        RoiIo.write(image, roi, rgb)
        rgb.release(); mask.release()
    }

    // ── Face makeup and facial hair ──────────────────────────────────────────

    private fun applyFace(image: Bitmap, parse: FaceParse, request: HairMakeupRequest, intensity: Float) {
        val roi = parse.roi ?: return
        val rgb = RoiIo.read(image, roi)
        val iod = parse.eyeDistance
        fun px(fraction: Float) = max(1, (iod * fraction).roundToInt())
        fun odd(fraction: Float) = px(fraction) * 2 + 1

        // Facial hair goes first so makeup (lips) sits on top of a moustache edge.
        if (request.facialHair != FacialHairStyle.NONE) {
            val color = request.facialHairColorHex?.let(::parseColor) ?: browColor(parse, image)
            facialHairAlpha(parse, request.facialHair)?.let { alphaCrop ->
                parse.project(alphaCrop, featherPx = px(0.01f))?.let { alpha ->
                    RegionEffects.tint(rgb, alpha, color, intensity, Blend.SHADED)
                    alpha.release()
                }
                alphaCrop.release()
            }
        }

        request.lipColorHex?.let { hex ->
            val lips = parse.classMask(FaceClass.U_LIP, FaceClass.L_LIP)
            parse.project(lips, featherPx = px(0.02f))?.let { mask ->
                val color = parseColor(hex)
                RegionEffects.recolor(
                    rgb, mask, strength = intensity * 0.85f, lightnessShift = 0.35f, chromaKeep = 0.3f,
                    keepHighlights = 0.6f,
                ) { color }
                mask.release()
            }
            lips.release()
        }

        val eyes = parse.classMask(FaceClass.L_EYE, FaceClass.R_EYE)
        request.eyeshadowColorHex?.let { hex ->
            // Lid band: eyes grown upward, minus the eye itself and the brows, clipped to skin.
            val grown = eyes.dilated(odd(0.17f))
            val lifted = grown.shifted(0.0, -iod * 0.09)
            val keepOut = eyes.dilated(odd(0.025f))
            val browsRaw = parse.classMask(FaceClass.L_BROW, FaceClass.R_BROW)
            val brows = browsRaw.dilated(odd(0.02f)).also { browsRaw.release() }
            val skin = parse.classMask(FaceClass.SKIN)
            val band = Mat()
            Core.bitwise_or(lifted, grown, band)
            Core.subtract(band, keepOut, band)
            Core.subtract(band, brows, band)
            Core.bitwise_and(band, skin, band)
            parse.project(band, featherPx = px(0.06f))?.let { mask ->
                RegionEffects.tint(rgb, mask, parseColor(hex), intensity * 0.7f, Blend.SOFT_LIGHT)
                RegionEffects.tint(rgb, mask, parseColor(hex), intensity * 0.25f, Blend.MULTIPLY)
                mask.release()
            }
            listOf(grown, lifted, keepOut, brows, skin, band).forEach { it.release() }
        }

        request.eyelinerColorHex?.let { hex ->
            // Upper contour: the eye shifted up a little, minus the eye.
            val up = eyes.shifted(0.0, -iod * 0.035)
            val line = Mat()
            Core.subtract(up, eyes, line)
            val thick = line.dilated(odd(0.006f))
            parse.project(thick, featherPx = px(0.008f))?.let { mask ->
                RegionEffects.tint(rgb, mask, parseColor(hex), intensity * 0.9f, Blend.NORMAL)
                mask.release()
            }
            listOf(up, line, thick).forEach { it.release() }
        }
        eyes.release()

        request.blushColorHex?.let { hex ->
            val blobs = Mat.zeros(parse.size, parse.size, CvType.CV_32FC1)
            for (p in listOf(parse.templatePoint(0.31f, 0.64f), parse.templatePoint(0.69f, 0.64f))) {
                Imgproc.ellipse(blobs, p, Size(iod * 0.30, iod * 0.20), 0.0, 0.0, 360.0, Scalar(1.0), -1)
            }
            val k = odd(0.22f).toDouble()
            Imgproc.GaussianBlur(blobs, blobs, Size(k, k), 0.0)
            val skin = parse.classMask(FaceClass.SKIN)
            val skinF = Mat().also { skin.convertTo(it, CvType.CV_32F, 1.0 / 255.0) }
            Core.multiply(blobs, skinF, blobs)
            parse.project(blobs, featherPx = px(0.03f))?.let { mask ->
                RegionEffects.tint(rgb, mask, parseColor(hex), intensity * 0.55f, Blend.SOFT_LIGHT)
                RegionEffects.tint(rgb, mask, parseColor(hex), intensity * 0.18f, Blend.MULTIPLY)
                mask.release()
            }
            listOf(blobs, skin, skinF).forEach { it.release() }
        }

        RoiIo.write(image, roi, rgb)
        rgb.release()
    }

    /**
     * Crop-space alpha (CV_32FC1, 0–1) of facial hair: a style region clipped to skin, filled
     * with downward-growing strand strokes over a base tint. Null when the region is empty.
     */
    private fun facialHairAlpha(parse: FaceParse, style: FacialHairStyle): Mat? {
        val k = parse.keypointsInCrop
        val iod = parse.eyeDistance
        val noseY = k[5].toDouble()
        val mouthL = Point(k[6].toDouble(), k[7].toDouble())
        val mouthR = Point(k[8].toDouble(), k[9].toDouble())
        val mouthY = (mouthL.y + mouthR.y) / 2
        val centerX = (k[0] + k[2] + k[4]) / 3.0
        val size = parse.size

        val region = Mat.zeros(size, size, CvType.CV_8UC1)
        fun rect(x0: Double, y0: Double, x1: Double, y1: Double) =
            Imgproc.rectangle(region, Point(x0, y0), Point(x1, y1), Scalar(255.0), -1)
        val lipTop = mouthY - iod * 0.10
        val mustacheTop = noseY + (mouthY - noseY) * 0.35
        when (style) {
            FacialHairStyle.MUSTACHE -> {
                rect(mouthL.x - iod * 0.12, mustacheTop, mouthR.x + iod * 0.12, lipTop)
            }
            FacialHairStyle.GOATEE -> {
                rect(mouthL.x - iod * 0.12, mustacheTop, mouthR.x + iod * 0.12, lipTop)
                rect(mouthL.x - iod * 0.02, mouthY, mouthR.x + iod * 0.02, size.toDouble())
            }
            FacialHairStyle.STUBBLE, FacialHairStyle.FULL_BEARD -> {
                rect(0.0, mustacheTop, size.toDouble(), size.toDouble())
                // Sideburns and cheek line: outer cheeks from the nose tip down.
                rect(0.0, noseY - iod * 0.15, centerX - iod * 0.72, size.toDouble())
                rect(centerX + iod * 0.72, noseY - iod * 0.15, size.toDouble(), size.toDouble())
            }
            FacialHairStyle.NONE -> Unit
        }
        val skin = parse.classMask(FaceClass.SKIN)
        Core.bitwise_and(region, skin, region)
        skin.release()
        if (Core.countNonZero(region) == 0) { region.release(); return null }

        val full = style == FacialHairStyle.FULL_BEARD || style == FacialHairStyle.GOATEE || style == FacialHairStyle.MUSTACHE
        val strands = Mat.zeros(size, size, CvType.CV_8UC1)
        val random = Random(SEED)
        val bounds = Imgproc.boundingRect(region)
        val count = (bounds.area() * if (full) 0.22 else 0.10).toInt()
        val maxLen = if (full) iod * 0.07 else iod * 0.025
        repeat(count) {
            val x = bounds.x + random.nextDouble() * bounds.width
            val y = bounds.y + random.nextDouble() * bounds.height
            val angle = PI / 2 + (random.nextDouble() - 0.5) * 1.1 // mostly downward
            val len = maxLen * (0.4 + 0.6 * random.nextDouble())
            Imgproc.line(
                strands, Point(x, y), Point(x + cos(angle) * len, y + sin(angle) * len),
                Scalar(160.0 + random.nextInt(96)), 1, Imgproc.LINE_AA,
            )
        }

        // alpha = region · (base + (1 − base) · strands) · opacity
        val baseFill = if (full) 0.45 else 0.12
        val opacity = if (full) 0.92 else 0.6
        val alpha = Mat()
        strands.convertTo(alpha, CvType.CV_32F, (1.0 - baseFill) / 255.0, baseFill)
        val regionF = Mat()
        region.convertTo(regionF, CvType.CV_32F, opacity / 255.0)
        // Soften the region edge so the beard thins out instead of ending in a hard line.
        val k2 = (max(3.0, iod * 0.12)).toInt() or 1
        Imgproc.GaussianBlur(regionF, regionF, Size(k2.toDouble(), k2.toDouble()), 0.0)
        Core.multiply(alpha, regionF, alpha)
        listOf(region, strands, regionF).forEach { it.release() }
        return alpha
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    /** Median-ish eyebrow colour, darkened slightly; a natural default for facial hair. */
    private fun browColor(parse: FaceParse, image: Bitmap): Int {
        val brows = parse.classMask(FaceClass.L_BROW, FaceClass.R_BROW)
        val pts = Mat()
        Core.findNonZero(brows, pts)
        brows.release()
        val n = pts.rows()
        if (n == 0) { pts.release(); return DEFAULT_FACIAL_HAIR }
        val coords = IntArray(n * 2).also { pts.get(0, 0, it) }
        pts.release()
        val lab = FloatArray(3)
        val ls = ArrayList<FloatArray>()
        val p = FloatArray(2)
        for (i in 0 until n step max(1, n / 400)) {
            p[0] = coords[i * 2].toFloat(); p[1] = coords[i * 2 + 1].toFloat()
            parse.cropToImage.mapPoints(p)
            val x = p[0].toInt().coerceIn(0, image.width - 1)
            val y = p[1].toInt().coerceIn(0, image.height - 1)
            ColorLab.toLab(image.getPixel(x, y) and 0xFFFFFF, lab)
            ls += lab.copyOf()
        }
        if (ls.isEmpty()) return DEFAULT_FACIAL_HAIR
        ls.sortBy { it[0] }
        val m = ls[ls.size / 3] // darker third: brow hairs, not the skin between them
        return ColorLab.toRgb((m[0] * 0.85f).coerceAtMost(40f), m[1], m[2])
    }

    /** First and last crop rows containing the mask. */
    private fun rowExtent(mask: Mat): Pair<Float, Float> {
        val r: Rect = Imgproc.boundingRect(mask)
        return if (r.height <= 0) 0f to mask.rows().toFloat() else r.y.toFloat() to (r.y + r.height).toFloat()
    }

    private fun parseColor(hex: String): Int = Color.parseColor("#$hex") and 0xFFFFFF

    private fun lighter(rgb: Int, deltaL: Float): Int {
        val lab = FloatArray(3).also { ColorLab.toLab(rgb, it) }
        return ColorLab.toRgb((lab[0] + deltaL).coerceAtMost(92f), lab[1] * 0.8f, lab[2] * 0.9f + 6f)
    }

    private fun lerpRgb(a: Int, b: Int, t: Float): Int {
        fun ch(shift: Int) = ((((a shr shift) and 0xFF) * (1 - t)) + (((b shr shift) and 0xFF) * t)).roundToInt()
        return (ch(16) shl 16) or (ch(8) shl 8) or ch(0)
    }

    private fun smooth(t: Float) = t * t * (3f - 2f * t)

    /** Smooth 1-D value noise in [0, 1]. */
    private fun valueNoise(x: Float): Float {
        val i = floor(x).toInt()
        val f = x - i
        fun hash(n: Int): Float {
            var h = n * 374761393 + 668265263
            h = (h xor (h ushr 13)) * 1274126177
            return ((h xor (h ushr 16)) and 0xFFFF) / 65535f
        }
        return hash(i) + (hash(i + 1) - hash(i)) * smooth(f)
    }

    private companion object {
        const val TAG = "HairMakeupEngine"
        const val MAX_FACES = 4
        const val SEED = 20261123
        const val DEFAULT_FACIAL_HAIR = 0x2B1E16
    }
}
