package com.offlinemorph.android.core.video

import android.graphics.Bitmap

/**
 * One decoded video frame.
 *
 * @param index              zero-based frame index in display order
 * @param bitmap             ARGB_8888 bitmap for this frame
 * @param presentationTimeUs presentation timestamp from the decoder (microseconds)
 */
data class IndexedFrame(
    val index: Int,
    val bitmap: Bitmap,
    val presentationTimeUs: Long,
)
