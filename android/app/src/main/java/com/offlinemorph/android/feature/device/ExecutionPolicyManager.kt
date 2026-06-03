package com.offlinemorph.android.feature.device

import android.app.ActivityManager
import android.content.Context
import android.os.PowerManager

/** The three quality tiers an engine can operate in. */
enum class QualityProfile { FAST, BALANCED, STUDIO }

/**
 * Resolved execution policy for the current device state.
 *
 * @param qualityProfile  the selected quality tier.
 * @param maxImageSizePx  longest-edge cap to apply before inference.
 * @param enhancerEnabled whether post-process enhancement (e.g. GFPGAN) is permitted.
 * @param reason          human-readable explanation of why this policy was selected.
 */
data class ExecutionPolicy(
    val qualityProfile: QualityProfile,
    val maxImageSizePx: Int,
    val enhancerEnabled: Boolean,
    val reason: String,
)

/**
 * Derives an [ExecutionPolicy] from real-time device state.
 *
 * Thermal status and memory pressure are checked first; static device-tier heuristics
 * apply only when the device is running cool with adequate memory available.
 *
 * Requires API 29 (minSdk = 29) for [PowerManager.currentThermalStatus].
 */
class ExecutionPolicyManager(private val context: Context) {

    fun currentPolicy(): ExecutionPolicy {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager

        if (pm.currentThermalStatus >= PowerManager.THERMAL_STATUS_MODERATE) {
            return ExecutionPolicy(
                qualityProfile = QualityProfile.FAST,
                maxImageSizePx = 512,
                enhancerEnabled = false,
                reason = "thermal pressure detected",
            )
        }

        if (mi.lowMemory) {
            return ExecutionPolicy(
                qualityProfile = QualityProfile.FAST,
                maxImageSizePx = 512,
                enhancerEnabled = false,
                reason = "low memory condition",
            )
        }

        return when (DeviceCapabilityAssessor(context).assess().tier) {
            "High"   -> ExecutionPolicy(QualityProfile.BALANCED, 1024, enhancerEnabled = true,  reason = "High-tier device")
            "Medium" -> ExecutionPolicy(QualityProfile.BALANCED, 768,  enhancerEnabled = false, reason = "Medium-tier device")
            else     -> ExecutionPolicy(QualityProfile.FAST,     512,  enhancerEnabled = false, reason = "Entry-tier device")
        }
    }
}
