package com.offlinemorph.android.feature.flags

/**
 * Build-time feature flags that gate in-development feature families.
 *
 * Set a flag to `true` locally to enable a feature that is under active development.
 * All flags default to `false` until the feature reaches beta-ready status and passes
 * the Definition of Done in IMPLEMENTATION_PLAN.md.
 */
object FeatureFlags {
    /** Phase B — Face swap quality uplift (boundary blending, multi-face, quality profiles). */
    val faceSwapUpliftEnabled: Boolean = false

    /** Phase C — AI Aging and De-aging synthesis. No public ONNX model available yet; hidden until one ships. */
    val agingEnabled: Boolean = false

    /** Phase C — Ancestry blend synthesis. */
    val ancestryEnabled: Boolean = false

    /** Phase D — Hair and Makeup studio. */
    val hairMakeupEnabled: Boolean = true

    /** Phase E — Beautifying and virtual plastic surgery controls. */
    val beautifyEnabled: Boolean = true

    /** Phase F — Cinematic 3D relighting. */
    val relightEnabled: Boolean = false

    /** Phase G — Background matting and scene replacement. */
    val backgroundSwapEnabled: Boolean = false
}
