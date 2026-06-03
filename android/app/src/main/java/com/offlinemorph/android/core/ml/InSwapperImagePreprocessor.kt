package com.offlinemorph.android.core.ml

import android.graphics.Bitmap
import kotlin.math.min

data class PreparedSwapTarget(
    val resizedBitmap: Bitmap,
    val inputTensor: FloatArray,
    val inputSize: Int,
)

class InSwapperImagePreprocessor {
    fun prepareTarget(bitmap: Bitmap, inputSize: Int = 128): PreparedSwapTarget {
        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val pixels = IntArray(inputSize * inputSize)
        resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        val tensor = FloatArray(1 * 3 * inputSize * inputSize)
        var pixelIndex = 0
        val channelStride = inputSize * inputSize

        repeat(inputSize) { y ->
            repeat(inputSize) { x ->
                val color = pixels[pixelIndex]
                val red = ((color shr 16) and 0xFF) / 255.0f
                val green = ((color shr 8) and 0xFF) / 255.0f
                val blue = (color and 0xFF) / 255.0f

                val offset = y * inputSize + x
                tensor[offset] = red
                tensor[channelStride + offset] = green
                tensor[channelStride * 2 + offset] = blue
                pixelIndex += 1
            }
        }

        return PreparedSwapTarget(
            resizedBitmap = resized,
            inputTensor = tensor,
            inputSize = inputSize,
        )
    }
}
