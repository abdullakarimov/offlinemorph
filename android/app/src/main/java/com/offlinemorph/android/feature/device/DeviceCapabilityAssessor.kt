package com.offlinemorph.android.feature.device

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import kotlin.math.roundToInt

class DeviceCapabilityAssessor(
    private val context: Context,
) {
    fun assess(): DeviceCapability {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)

        val totalMemGb = memoryInfo.totalMem.toDouble() / (1024.0 * 1024.0 * 1024.0)
        val cpuCores = Runtime.getRuntime().availableProcessors()

        val tier = when {
            totalMemGb >= 10.0 && cpuCores >= 8 -> "High"
            totalMemGb >= 6.0 && cpuCores >= 6 -> "Medium"
            else -> "Entry"
        }

        val recommendation = when (tier) {
            "High" -> "Suitable for on-device face swapping workloads. Focus on model compatibility and optimization rather than raw device power."
            "Medium" -> "Usable for on-device swapping with moderate image sizes. Prefer 512-1024px processing and memory-aware batching."
            else -> "May run model checks and lighter inference, but full-quality swaps could be slow or memory-constrained."
        }

        return DeviceCapability(
            tier = tier,
            summary = "${Build.MANUFACTURER} ${Build.MODEL} | Android ${Build.VERSION.RELEASE} | RAM ${totalMemGb.roundToInt()} GB | CPU cores $cpuCores",
            recommendation = recommendation,
        )
    }
}
