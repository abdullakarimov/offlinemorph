package com.offlinemorph.android.core.image

import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.content.ContentResolver
import kotlin.math.max
import kotlin.math.roundToInt

data class LoadedBitmap(
    val bitmap: Bitmap,
    val width: Int,
    val height: Int,
)

interface BitmapLoader {
    /**
     * Decodes [uri] into a software ARGB_8888 bitmap. When [maxDimensionPx] > 0 the image is
     * downsampled during decode so its longest edge does not exceed that value, which avoids
     * materialising full-resolution 50–200 MP camera images in memory.
     */
    fun load(uri: Uri, maxDimensionPx: Int = 0): LoadedBitmap
}

class AndroidBitmapLoader(
    private val contentResolver: ContentResolver,
) : BitmapLoader {
    override fun load(uri: Uri, maxDimensionPx: Int): LoadedBitmap {
        val source = ImageDecoder.createSource(contentResolver, uri)
        val bitmap = ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
            decoder.isMutableRequired = false
            // Software pixels are required by getPixels() and OpenCV's bitmapToMat().
            decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE)
            val longest = max(info.size.width, info.size.height)
            if (maxDimensionPx in 1 until longest) {
                val scale = maxDimensionPx.toFloat() / longest
                decoder.setTargetSize(
                    (info.size.width * scale).roundToInt().coerceAtLeast(1),
                    (info.size.height * scale).roundToInt().coerceAtLeast(1),
                )
            }
        }.let { decoded ->
            // HDR / wide-gamut sources can decode as RGBA_F16; the ML pipeline expects 8-bit.
            if (decoded.config == Bitmap.Config.ARGB_8888) decoded
            else decoded.copy(Bitmap.Config.ARGB_8888, false).also { decoded.recycle() }
        }
        return LoadedBitmap(bitmap = bitmap, width = bitmap.width, height = bitmap.height)
    }
}
