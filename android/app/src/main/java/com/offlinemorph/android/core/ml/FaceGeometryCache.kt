package com.offlinemorph.android.core.ml

import android.graphics.Bitmap

/**
 * Thread-safe LRU cache for [FaceAnalysisSummary] results.
 *
 * Keyed by [Bitmap] identity (not pixel content) so that the same bitmap instance reused
 * across multiple feature engines in a single session avoids redundant face-detection and
 * landmark passes.
 *
 * @param maxEntries maximum number of entries kept before the oldest is evicted.
 */
class FaceGeometryCache(private val maxEntries: Int = 8) {

    private val store = object : LinkedHashMap<Int, FaceAnalysisSummary>(
        maxEntries + 1, 0.75f, /* accessOrder= */ true,
    ) {
        override fun removeEldestEntry(
            eldest: MutableMap.MutableEntry<Int, FaceAnalysisSummary>,
        ) = size > maxEntries
    }

    private val lock = Any()

    /** Returns the cached analysis for [bitmap], or null if not present. */
    fun get(bitmap: Bitmap): FaceAnalysisSummary? = synchronized(lock) {
        store[System.identityHashCode(bitmap)]
    }

    /** Stores [summary] under [bitmap]'s identity. */
    fun put(bitmap: Bitmap, summary: FaceAnalysisSummary): Unit = synchronized(lock) {
        store[System.identityHashCode(bitmap)] = summary
    }

    /** Removes the cached entry for [bitmap] if present. */
    fun invalidate(bitmap: Bitmap): Unit = synchronized(lock) {
        store.remove(System.identityHashCode(bitmap))
    }

    /** Evicts all cached entries. */
    fun clear(): Unit = synchronized(lock) { store.clear() }
}
