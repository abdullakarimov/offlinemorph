package com.offlinemorph.android.core.ml

import android.graphics.Bitmap

class BitmapTensorConverter {
    /**
     * Converts a Bitmap to a CHW float32 tensor.
     *
     * Formula per channel: (pixelValue / scaleDivisor - mean[c]) / std[c]
     *
     * @param swapChannels When true the R and B planes are swapped in the output,
     *   producing BGR-CHW rather than RGB-CHW.  Required by SCRFD detectors which
     *   were trained on OpenCV BGR input.
     */
    fun toNormalizedChwFloatArray(
        bitmap: Bitmap,
        mean: FloatArray = floatArrayOf(0f, 0f, 0f),
        std: FloatArray = floatArrayOf(1f, 1f, 1f),
        scaleDivisor: Float = 255.0f,
        swapChannels: Boolean = false,
    ): FloatArray {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val tensor = FloatArray(3 * width * height)
        val stride = width * height

        if (!swapChannels) {
            pixels.forEachIndexed { index, color ->
                val red = ((color shr 16) and 0xFF) / scaleDivisor
                val green = ((color shr 8) and 0xFF) / scaleDivisor
                val blue = (color and 0xFF) / scaleDivisor

                tensor[index] = (red - mean[0]) / std[0]
                tensor[stride + index] = (green - mean[1]) / std[1]
                tensor[stride * 2 + index] = (blue - mean[2]) / std[2]
            }
        } else {
            // BGR-CHW: plane 0 = Blue, plane 1 = Green, plane 2 = Red
            pixels.forEachIndexed { index, color ->
                val red = ((color shr 16) and 0xFF) / scaleDivisor
                val green = ((color shr 8) and 0xFF) / scaleDivisor
                val blue = (color and 0xFF) / scaleDivisor

                tensor[index] = (blue - mean[0]) / std[0]
                tensor[stride + index] = (green - mean[1]) / std[1]
                tensor[stride * 2 + index] = (red - mean[2]) / std[2]
            }
        }

        return tensor
    }
}
