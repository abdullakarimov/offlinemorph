package com.offlinemorph.android.core.ml.relight

import android.graphics.Bitmap
import com.offlinemorph.android.core.ml.EngineProgress
import com.offlinemorph.android.core.ml.EngineResult

/**
 * One virtual light in the relighting rig.
 *
 * @param azimuthDegrees   horizontal angle around the subject; 0 = frontal, +90 = from the subject's left.
 * @param elevationDegrees vertical angle; 0 = eye level, +90 = directly overhead.
 * @param intensity        light strength in [0f, 2f]; 1.0 is neutral.
 * @param softness         0 = hard point light with crisp shadows, 1 = large diffuse source.
 * @param colorArgb        light colour as packed ARGB (alpha ignored).
 */
data class VirtualLight(
    val azimuthDegrees: Float,
    val elevationDegrees: Float,
    val intensity: Float = 1.0f,
    val softness: Float = 0.5f,
    val colorArgb: Int = 0xFFFFFFFF.toInt(),
)

/**
 * Input request for the cinematic relighting engine (Phase F).
 *
 * @param sourceBitmap    portrait to relight.
 * @param lights          virtual light rig; at least one light. Applied additively over [ambient].
 * @param ambient         ambient fill level in [0f, 1f] retained from the original lighting.
 * @param preserveSkinTone when true, the albedo estimate keeps the subject's skin colour stable.
 */
data class RelightRequest(
    val sourceBitmap: Bitmap,
    val lights: List<VirtualLight>,
    val ambient: Float = 0.3f,
    val preserveSkinTone: Boolean = true,
)

/**
 * Output produced by a successful relight run.
 *
 * @param outputBitmap the relit portrait.
 * @param normalMap    optional per-pixel surface-normal visualisation (RGB-encoded), for debugging
 *                     and for re-rendering cheaply while the user drags a light.
 */
data class RelightResult(
    val outputBitmap: Bitmap,
    val normalMap: Bitmap? = null,
)

/** Contract for on-device relighting engines. */
interface RelightEngine {
    suspend fun runRelight(
        request: RelightRequest,
        onProgress: (EngineProgress) -> Unit = {},
    ): EngineResult<RelightResult>
}
