package com.offlinemorph.android.core.ml.face

import kotlin.math.cbrt
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/** sRGB ⇄ CIE Lab (D65), L in [0, 100]. */
object ColorLab {
    private val toLinear = FloatArray(256) { i ->
        val c = i / 255f
        if (c <= 0.04045f) c / 12.92f else ((c + 0.055f) / 1.055f).toDouble().pow(2.4).toFloat()
    }

    private fun f(t: Float): Float = if (t > 0.008856f) cbrt(t) else 7.787f * t + 16f / 116f

    /** Writes (L, a, b) of the 8-bit sRGB colour into [out]. */
    fun toLab(r: Int, g: Int, b: Int, out: FloatArray) {
        val lr = toLinear[r]; val lg = toLinear[g]; val lb = toLinear[b]
        val x = (0.4124f * lr + 0.3576f * lg + 0.1805f * lb) / 0.95047f
        val y = 0.2126f * lr + 0.7152f * lg + 0.0722f * lb
        val z = (0.0193f * lr + 0.1192f * lg + 0.9505f * lb) / 1.08883f
        val fx = f(x); val fy = f(y); val fz = f(z)
        out[0] = 116f * fy - 16f
        out[1] = 500f * (fx - fy)
        out[2] = 200f * (fy - fz)
    }

    fun toLab(rgb: Int, out: FloatArray) = toLab((rgb shr 16) and 0xFF, (rgb shr 8) and 0xFF, rgb and 0xFF, out)

    private fun fInv(t: Float): Float = if (t > 0.206893f) t * t * t else (t - 16f / 116f) / 7.787f

    private fun toSrgb8(c: Float): Int {
        val v = if (c <= 0.0031308f) 12.92f * c else 1.055f * c.toDouble().pow(1 / 2.4).toFloat() - 0.055f
        return (v * 255f + 0.5f).toInt().coerceIn(0, 255)
    }

    /** Lab → packed 0xRRGGBB. */
    fun toRgb(l: Float, a: Float, b: Float): Int {
        val fy = (l + 16f) / 116f
        val x = 0.95047f * fInv(fy + a / 500f)
        val y = fInv(fy)
        val z = 1.08883f * fInv(fy - b / 200f)
        val r = toSrgb8(3.2406f * x - 1.5372f * y - 0.4986f * z)
        val g = toSrgb8(-0.9689f * x + 1.8758f * y + 0.0415f * z)
        val bl = toSrgb8(0.0557f * x - 0.2040f * y + 1.0570f * z)
        return (r shl 16) or (g shl 8) or bl
    }
}

/**
 * Pixel effects applied inside a face ROI. Every function takes the ROI as an RGB `CV_8UC3`
 * [Mat] (modified in place) and a `CV_32FC1` weight mask of the same size in [0, 1].
 */
object RegionEffects {

    enum class Blend { NORMAL, MULTIPLY, SOFT_LIGHT, SHADED }

    private fun smoothstep(e0: Float, e1: Float, x: Float): Float {
        val t = ((x - e0) / (e1 - e0)).coerceIn(0f, 1f)
        return t * t * (3f - 2f * t)
    }

    private class Buffers(rgb: Mat, mask: Mat) {
        val n = rgb.rows() * rgb.cols()
        val px = ByteArray(n * 3).also { rgb.get(0, 0, it) }
        val w = FloatArray(n).also { mask.get(0, 0, it) }
        fun r(i: Int) = px[i * 3].toInt() and 0xFF
        fun g(i: Int) = px[i * 3 + 1].toInt() and 0xFF
        fun b(i: Int) = px[i * 3 + 2].toInt() and 0xFF
        fun mix(i: Int, rgb: Int, t: Float) {
            val k = t.coerceIn(0f, 1f)
            px[i * 3] = (r(i) + (((rgb shr 16) and 0xFF) - r(i)) * k).roundToInt().toByte()
            px[i * 3 + 1] = (g(i) + (((rgb shr 8) and 0xFF) - g(i)) * k).roundToInt().toByte()
            px[i * 3 + 2] = (b(i) + ((rgb and 0xFF) - b(i)) * k).roundToInt().toByte()
        }
        fun commit(rgb: Mat) = rgb.put(0, 0, px)
    }

    /**
     * Recolours the masked region in Lab space. Lightness keeps the region's own shading and
     * shifts its mean by [lightnessShift] × (target − mean); chroma moves to the target while
     * keeping [chromaKeep] of the region's own variation.
     *
     * @param targetAt per-pixel target colour (0xRRGGBB) for gradients and streaks; index = row × cols + col.
     * @param keepHighlights 0..1: how much bright specular pixels (gloss) resist the change.
     */
    fun recolor(
        rgb: Mat,
        mask: Mat,
        strength: Float,
        lightnessShift: Float,
        chromaKeep: Float,
        keepHighlights: Float = 0f,
        targetAt: (Int) -> Int,
    ) {
        val buf = Buffers(rgb, mask)
        val lab = FloatArray(3)
        var sumW = 0.0; var sumL = 0.0; var sumA = 0.0; var sumB = 0.0
        for (i in 0 until buf.n) {
            val w = buf.w[i]
            if (w < 0.3f) continue
            ColorLab.toLab(buf.r(i), buf.g(i), buf.b(i), lab)
            sumW += w; sumL += w * lab[0]; sumA += w * lab[1]; sumB += w * lab[2]
        }
        if (sumW <= 0.0) return
        val meanL = (sumL / sumW).toFloat(); val meanA = (sumA / sumW).toFloat(); val meanB = (sumB / sumW).toFloat()

        val target = FloatArray(3)
        var cachedTarget = Int.MIN_VALUE
        for (i in 0 until buf.n) {
            var w = buf.w[i] * strength
            if (w < 0.004f) continue
            ColorLab.toLab(buf.r(i), buf.g(i), buf.b(i), lab)
            if (keepHighlights > 0f) w *= 1f - keepHighlights * smoothstep(meanL + 15f, meanL + 35f, lab[0])
            val t = targetAt(i)
            if (t != cachedTarget) { ColorLab.toLab(t, target); cachedTarget = t }
            val l = (lab[0] + (target[0] - meanL) * lightnessShift).coerceIn(0f, 100f)
            val a = target[1] + (lab[1] - meanA) * chromaKeep
            val b = target[2] + (lab[2] - meanB) * chromaKeep
            buf.mix(i, ColorLab.toRgb(l, a, b), w)
        }
        buf.commit(rgb)
    }

    /**
     * Whitens teeth: inside the mask, only bright, low-chroma pixels (tooth-like) are changed —
     * yellow (b*) and red (a*) are reduced and lightness is lifted a little.
     */
    fun whitenTeeth(rgb: Mat, mask: Mat, strength: Float) {
        val buf = Buffers(rgb, mask)
        val lab = FloatArray(3)
        for (i in 0 until buf.n) {
            val w0 = buf.w[i] * strength
            if (w0 < 0.004f) continue
            ColorLab.toLab(buf.r(i), buf.g(i), buf.b(i), lab)
            val chroma = hypot(lab[1], lab[2])
            val toothLike = smoothstep(35f, 65f, lab[0]) * (1f - smoothstep(28f, 48f, chroma))
            val w = (w0 * toothLike).coerceIn(0f, 1f)
            if (w < 0.004f) continue
            val l = lab[0] + (100f - lab[0]) * 0.22f * w
            val a = lab[1] * (1f - 0.5f * w)
            val b = lab[2] * (1f - 0.8f * w)
            buf.mix(i, ColorLab.toRgb(l, a, b), 1f)
        }
        buf.commit(rgb)
    }

    /** Blends a flat [color] (0xRRGGBB) over the mask with opacity [alpha]. */
    fun tint(rgb: Mat, mask: Mat, color: Int, alpha: Float, blend: Blend) {
        val buf = Buffers(rgb, mask)
        val cr = ((color shr 16) and 0xFF) / 255f
        val cg = ((color shr 8) and 0xFF) / 255f
        val cb = (color and 0xFF) / 255f
        for (i in 0 until buf.n) {
            val w = buf.w[i] * alpha
            if (w < 0.004f) continue
            val sr = buf.r(i) / 255f; val sg = buf.g(i) / 255f; val sb = buf.b(i) / 255f
            val out = when (blend) {
                Blend.NORMAL -> floatArrayOf(cr, cg, cb)
                Blend.MULTIPLY -> floatArrayOf(sr * cr, sg * cg, sb * cb)
                Blend.SOFT_LIGHT -> floatArrayOf(softLight(sr, cr), softLight(sg, cg), softLight(sb, cb))
                Blend.SHADED -> {
                    // Colour shaded by the underlying luminance, so strands pick up skin shading.
                    val lum = 0.55f + 0.9f * (0.299f * sr + 0.587f * sg + 0.114f * sb)
                    floatArrayOf(min(1f, cr * lum), min(1f, cg * lum), min(1f, cb * lum))
                }
            }
            val packed = ((out[0] * 255f).roundToInt().coerceIn(0, 255) shl 16) or
                ((out[1] * 255f).roundToInt().coerceIn(0, 255) shl 8) or
                (out[2] * 255f).roundToInt().coerceIn(0, 255)
            buf.mix(i, packed, w)
        }
        buf.commit(rgb)
    }

    /** Pegtop soft light: (1 − 2b)·a² + 2b·a. */
    private fun softLight(a: Float, b: Float) = (1f - 2f * b) * a * a + 2f * b * a

    /**
     * Skin retouch by frequency separation: an edge-preserving (bilateral) filter is run at a
     * resolution where the eyes are ~[WORK_IOD] px apart, and only that low-frequency
     * correction is added back at full resolution. Blemishes and blotchy tone even out while
     * pores and fine texture survive, which avoids the plastic look of a full-resolution blur.
     *
     * @param eyeDistancePx inter-ocular distance in ROI pixels; sets the working scale.
     */
    fun smoothSkin(rgb: Mat, mask: Mat, strength: Float, eyeDistancePx: Float) {
        if (strength <= 0f) return
        val scale = (WORK_IOD / max(eyeDistancePx, 1f)).coerceAtMost(1f)
        val small = Mat()
        val smallSize = Size(max(8.0, rgb.cols() * scale.toDouble()), max(8.0, rgb.rows() * scale.toDouble()))
        Imgproc.resize(rgb, small, smallSize, 0.0, 0.0, Imgproc.INTER_AREA)

        var filtered = Mat()
        val sigmaColor = 18.0 + 32.0 * strength
        Imgproc.bilateralFilter(small, filtered, 9, sigmaColor, 6.0)
        if (strength > 0.5f) {
            val second = Mat()
            Imgproc.bilateralFilter(filtered, second, 9, sigmaColor, 6.0)
            filtered.release()
            filtered = second
        }

        // correction = filtered − small, upsampled.
        val smallF = Mat().also { small.convertTo(it, CvType.CV_32FC3) }
        val correction = Mat().also { filtered.convertTo(it, CvType.CV_32FC3) }
        org.opencv.core.Core.subtract(correction, smallF, correction)
        val correctionFull = Mat()
        Imgproc.resize(correction, correctionFull, rgb.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)

        val n = rgb.rows() * rgb.cols()
        val px = ByteArray(n * 3).also { rgb.get(0, 0, it) }
        val corr = FloatArray(n * 3).also { correctionFull.get(0, 0, it) }
        val w = FloatArray(n).also { mask.get(0, 0, it) }
        val k = strength.coerceIn(0f, 1f)
        for (i in 0 until n) {
            val m = w[i] * k
            if (m < 0.004f) continue
            for (c in 0 until 3) {
                val j = i * 3 + c
                px[j] = ((px[j].toInt() and 0xFF) + corr[j] * m).roundToInt().coerceIn(0, 255).toByte()
            }
        }
        rgb.put(0, 0, px)
        listOf(small, filtered, smallF, correction, correctionFull).forEach { it.release() }
    }

    /** Inter-ocular distance, in pixels, at which skin smoothing works. */
    private const val WORK_IOD = 110f
}

/**
 * Accumulates smooth inverse-mapping deformations over an ROI and applies them in one remap.
 * Each operation is capped so edits stay natural and the map stays one-to-one; displacements
 * are summed, which is accurate because the regions (eyes, cheeks) barely overlap.
 */
class FaceWarp(private val width: Int, private val height: Int) {
    private val dx = FloatArray(width * height)
    private val dy = FloatArray(width * height)
    var isEmpty = true
        private set

    /**
     * Magnifies a disc of radius [r] around ([cx], [cy]); [amount] in [0, MAX_BULGE] is the
     * relative shrink of the sampling radius at the centre.
     */
    fun bulge(cx: Float, cy: Float, r: Float, amount: Float) {
        val a = amount.coerceIn(0f, MAX_BULGE)
        if (a <= 0f || r < 2f) return
        forDisc(cx, cy, r) { i, ox, oy, d2 ->
            val k = a * (1f - d2 / (r * r))
            dx[i] -= ox * k
            dy[i] -= oy * k
        }
    }

    /**
     * Adds a displacement field over the whole ROI: [field] receives (x, y) and writes the
     * inverse-map offset (dx, dy) into its 2-float output. The field must be smooth with a
     * displacement gradient well below 1 so the warp cannot fold over.
     */
    fun addField(field: (x: Int, y: Int, out: FloatArray) -> Unit) {
        val d = FloatArray(2)
        for (y in 0 until height) for (x in 0 until width) {
            d[0] = 0f; d[1] = 0f
            field(x, y, d)
            if (d[0] != 0f || d[1] != 0f) {
                val i = y * width + x
                dx[i] += d[0]; dy[i] += d[1]
                isEmpty = false
            }
        }
    }

    private inline fun forDisc(cx: Float, cy: Float, r: Float, op: (Int, Float, Float, Float) -> Unit) {
        val x0 = max(0, (cx - r).toInt()); val x1 = min(width - 1, (cx + r).toInt() + 1)
        val y0 = max(0, (cy - r).toInt()); val y1 = min(height - 1, (cy + r).toInt() + 1)
        val r2 = r * r
        for (y in y0..y1) for (x in x0..x1) {
            val ox = x - cx; val oy = y - cy
            val d2 = ox * ox + oy * oy
            if (d2 < r2) { op(y * width + x, ox, oy, d2); isEmpty = false }
        }
    }

    /** Returns the warped image (new Mat); [rgb] is untouched. Caller releases. */
    fun apply(rgb: Mat): Mat {
        val mapX = Mat(height, width, CvType.CV_32FC1)
        val mapY = Mat(height, width, CvType.CV_32FC1)
        val bx = FloatArray(width * height)
        val by = FloatArray(width * height)
        for (y in 0 until height) for (x in 0 until width) {
            val i = y * width + x
            bx[i] = x + dx[i]; by[i] = y + dy[i]
        }
        mapX.put(0, 0, bx); mapY.put(0, 0, by)
        val out = Mat()
        Imgproc.remap(rgb, out, mapX, mapY, Imgproc.INTER_LINEAR, org.opencv.core.Core.BORDER_REPLICATE)
        mapX.release(); mapY.release()
        return out
    }

    companion object {
        const val MAX_BULGE = 0.3f
    }
}
