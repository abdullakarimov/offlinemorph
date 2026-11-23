package com.offlinemorph.android.feature.swap

import com.offlinemorph.android.feature.device.QualityProfile

/**
 * Deterministic swap settings for a [QualityProfile]. The same profile always produces the
 * same pipeline, so results are reproducible; the device policy can only lower the profile.
 *
 * | Profile  | Pixel boost | Poisson blend | Occlusion mask | GFPGAN | Max image px |
 * |----------|-------------|---------------|----------------|--------|--------------|
 * | Fast     | 1×          | no            | no             | no     | 1280         |
 * | Balanced | 1×          | yes           | yes            | no     | 2048         |
 * | Studio   | 2×          | yes           | yes            | yes    | 4096         |
 */
data class SwapQualitySettings(
    val pixelBoost: Int,
    val seamlessBlend: Boolean,
    val occlusionAware: Boolean,
    val enhancer: Boolean,
    val maxImageSizePx: Int,
)

fun QualityProfile.swapSettings(): SwapQualitySettings = when (this) {
    QualityProfile.FAST -> SwapQualitySettings(1, seamlessBlend = false, occlusionAware = false, enhancer = false, maxImageSizePx = 1280)
    QualityProfile.BALANCED -> SwapQualitySettings(1, seamlessBlend = true, occlusionAware = true, enhancer = false, maxImageSizePx = 2048)
    QualityProfile.STUDIO -> SwapQualitySettings(2, seamlessBlend = true, occlusionAware = true, enhancer = true, maxImageSizePx = 4096)
}

val QualityProfile.displayName: String
    get() = when (this) {
        QualityProfile.FAST -> "Fast"
        QualityProfile.BALANCED -> "Balanced"
        QualityProfile.STUDIO -> "Studio"
    }

val QualityProfile.description: String
    get() = when (this) {
        QualityProfile.FAST -> "Quickest. Feathered blend, no extra passes."
        QualityProfile.BALANCED -> "Colour-matched blend that keeps hair, hands and glasses in front of the face."
        QualityProfile.STUDIO -> "Balanced plus 2× detail and AI face restoration (GFPGAN, if installed). ~4–6× slower."
    }

/** The lower of the user's choice and what the device's thermal / memory policy allows. */
fun effectiveProfile(user: QualityProfile, policy: QualityProfile): QualityProfile =
    if (user.ordinal <= policy.ordinal) user else policy
