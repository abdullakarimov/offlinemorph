package com.offlinemorph.android.core.ml

import android.graphics.Bitmap
import android.graphics.Canvas

data class PreparedSquareCrop(
    val bitmap: Bitmap,
    val scale: Float,
)

class SquareCropPreprocessor {
    fun prepare(bitmap: Bitmap, outputSize: Int): PreparedSquareCrop {
        val srcWidth = bitmap.width
        val srcHeight = bitmap.height

        val (destWidth, destHeight, scale) = if (srcHeight > srcWidth) {
            val height = outputSize
            val width = (srcWidth.toFloat() / srcHeight.toFloat() * outputSize).toInt().coerceAtLeast(1)
            Triple(width, height, outputSize.toFloat() / srcHeight.toFloat())
        } else {
            val width = outputSize
            val height = (srcHeight.toFloat() / srcWidth.toFloat() * outputSize).toInt().coerceAtLeast(1)
            Triple(width, height, outputSize.toFloat() / srcWidth.toFloat())
        }

        val resized = Bitmap.createScaledBitmap(bitmap, destWidth, destHeight, true)
        val outputBitmap = Bitmap.createBitmap(outputSize, outputSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(outputBitmap)
        canvas.drawBitmap(resized, 0f, 0f, null)

        return PreparedSquareCrop(
            bitmap = outputBitmap,
            scale = scale,
        )
    }
}
