package com.offlinemorph.android.core.ml

import android.graphics.Bitmap
import java.lang.ref.WeakReference

/**
 * Thread-safe LRU cache for [FaceAnalysisSummary] results, keyed by [Bitmap] instance.
 *
 * Entries hold the bitmap weakly and are verified by reference and generation id on lookup,
 * so a recycled video frame whose identity hash is reused by a new frame can never return
 * another frame's faces.
 *
 * @param maxEntries maximum number of entries kept before the oldest is evicted.
 */
class FaceGeometryCache(private val maxEntries: Int = 8) {

    private class Entry(
        val bitmap: WeakReference<Bitmap>,
        val generationId: Int,
        val summary: FaceAnalysisSummary,
        val hasEmbedding: Boolean,
        val hasGender: Boolean,
    )

    private val store = object : LinkedHashMap<Int, Entry>(
        maxEntries + 1, 0.75f, /* accessOrder= */ true,
    ) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<Int, Entry>) = size > maxEntries
    }

    private val lock = Any()

    /** Returns the cached analysis for [bitmap] if it was computed with at least the requested outputs. */
    fun get(bitmap: Bitmap, needEmbedding: Boolean, needGender: Boolean): FaceAnalysisSummary? = synchronized(lock) {
        val entry = store[System.identityHashCode(bitmap)] ?: return null
        val valid = entry.bitmap.get() === bitmap &&
            entry.generationId == bitmap.generationId &&
            (entry.hasEmbedding || !needEmbedding) &&
            (entry.hasGender || !needGender)
        if (valid) entry.summary else null
    }

    fun put(bitmap: Bitmap, summary: FaceAnalysisSummary, hasEmbedding: Boolean, hasGender: Boolean): Unit = synchronized(lock) {
        store[System.identityHashCode(bitmap)] =
            Entry(WeakReference(bitmap), bitmap.generationId, summary, hasEmbedding, hasGender)
    }
}
