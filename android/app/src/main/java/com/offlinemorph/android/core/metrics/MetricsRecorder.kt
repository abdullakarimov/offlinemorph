package com.offlinemorph.android.core.metrics

import android.util.Log
import java.io.File
import java.time.Instant

/** Records a single engine latency observation. No user media is ever captured here. */
data class LatencyEvent(
    val feature: String,
    val phase: String,
    val durationMs: Long,
    val timestamp: Instant = Instant.now(),
)

/** Records a memory peak observation for a feature pipeline run. */
data class MemoryEvent(
    val feature: String,
    val peakBytesUsed: Long,
    val timestamp: Instant = Instant.now(),
)

/**
 * Local-only metrics sink.
 *
 * Writes structured latency, memory-peak, and thermal-throttle events to logcat and
 * optionally to an append-only flat log file on-device. No user media content is
 * captured, transmitted, or referenced at any point.
 *
 * @param logFile optional on-device file for persistent metric storage. Pass null to
 *                limit output to logcat only.
 */
class MetricsRecorder(private val logFile: File? = null) {

    private val tag = "OFFLINEMORPH_METRICS"

    fun recordLatency(event: LatencyEvent) {
        val msg = "LATENCY feature=${event.feature} phase=${event.phase} durationMs=${event.durationMs}"
        Log.d(tag, msg)
        logFile?.appendText("${event.timestamp} $msg\n")
    }

    fun recordMemoryPeak(event: MemoryEvent) {
        val peakMb = event.peakBytesUsed / (1024L * 1024L)
        val msg = "MEMORY feature=${event.feature} peakMb=$peakMb"
        Log.d(tag, msg)
        logFile?.appendText("${event.timestamp} $msg\n")
    }

    fun recordThermalThrottle(feature: String) {
        val msg = "THERMAL_THROTTLE feature=$feature"
        Log.w(tag, msg)
        logFile?.appendText("${Instant.now()} $msg\n")
    }

    /**
     * Executes [block], records its wall-clock duration as a [LatencyEvent], and returns the
     * block result.
     */
    inline fun <T> measureLatency(feature: String, phase: String, block: () -> T): T {
        val start = System.currentTimeMillis()
        val result = block()
        recordLatency(LatencyEvent(feature, phase, System.currentTimeMillis() - start))
        return result
    }
}
