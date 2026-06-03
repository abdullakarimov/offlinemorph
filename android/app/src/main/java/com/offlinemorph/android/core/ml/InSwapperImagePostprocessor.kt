package com.offlinemorph.android.core.ml

import android.graphics.Bitmap
import kotlin.math.roundToInt

class InSwapperImagePostprocessor {
    fun toBitmap(outputTensor: FloatArray, width: Int, height: Int): Bitmap {
        require(outputTensor.size == width * height * 3) {
            "Expected output tensor size ${width * height * 3}, got ${outputTensor.size}"
        }

        val pixels = IntArray(width * height)
        val channelStride = width * height

        for (index in pixels.indices) {
            val red = (outputTensor[index] * 255.0f).toInt().coerceIn(0, 255)
            val green = (outputTensor[channelStride + index] * 255.0f).toInt().coerceIn(0, 255)
            val blue = (outputTensor[channelStride * 2 + index] * 255.0f).toInt().coerceIn(0, 255)
            pixels[index] = (255 shl 24) or (red shl 16) or (green shl 8) or blue
        }

        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }

    fun toBitmapNhwc(outputTensor: FloatArray, width: Int, height: Int): Bitmap {
        require(outputTensor.size == width * height * 3) {
            "Expected output tensor size ${width * height * 3}, got ${outputTensor.size}"
        }

        val pixels = IntArray(width * height)
        var idx = 0
        for (i in pixels.indices) {
            val red = (outputTensor[idx++] * 255.0f).toInt().coerceIn(0, 255)
            val green = (outputTensor[idx++] * 255.0f).toInt().coerceIn(0, 255)
            val blue = (outputTensor[idx++] * 255.0f).toInt().coerceIn(0, 255)
            pixels[i] = (255 shl 24) or (red shl 16) or (green shl 8) or blue
        }

        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }

    fun toBitmapFromTensor(output: TensorData, expectedSize: Int): Bitmap? {
        val expectedFlat = expectedSize * expectedSize * 3
        if (output.data.size != expectedFlat) {
            return null
        }

        val shape = output.shape
        if (shape != null && shape.size == 4) {
            val cSecond = shape[1] == 3L
            val cLast = shape[3] == 3L
            val h = shape[2].toInt()
            if (cSecond && h == expectedSize && shape[3].toInt() == expectedSize) {
                return toBitmap(output.data, expectedSize, expectedSize)
            }

            val nhwcH = shape[1].toInt()
            val nhwcW = shape[2].toInt()
            if (cLast && nhwcH == expectedSize && nhwcW == expectedSize) {
                return toBitmapNhwc(output.data, expectedSize, expectedSize)
            }
        }

        // Fallback to NCHW interpretation.
        return toBitmap(output.data, expectedSize, expectedSize)
    }

    fun createPreviewBitmap(bitmap: Bitmap, maxEdge: Int = 512): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        val longestEdge = maxOf(width, height)
        if (longestEdge <= maxEdge) {
            return bitmap
        }

        val scale = maxEdge.toFloat() / longestEdge.toFloat()
        val outWidth = (width * scale).roundToInt().coerceAtLeast(1)
        val outHeight = (height * scale).roundToInt().coerceAtLeast(1)
        return Bitmap.createScaledBitmap(bitmap, outWidth, outHeight, true)
    }
}
