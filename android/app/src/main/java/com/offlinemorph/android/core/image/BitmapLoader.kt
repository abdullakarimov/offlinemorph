package com.offlinemorph.android.core.image

import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.content.ContentResolver

data class LoadedBitmap(
    val bitmap: Bitmap,
    val width: Int,
    val height: Int,
)

interface BitmapLoader {
    fun load(uri: Uri): LoadedBitmap
}

class AndroidBitmapLoader(
    private val contentResolver: ContentResolver,
) : BitmapLoader {
    override fun load(uri: Uri): LoadedBitmap {
        val source = ImageDecoder.createSource(contentResolver, uri)
        val bitmap = ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
            decoder.isMutableRequired = false
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE)
            }
        }
        return LoadedBitmap(bitmap = bitmap, width = bitmap.width, height = bitmap.height)
    }
}
