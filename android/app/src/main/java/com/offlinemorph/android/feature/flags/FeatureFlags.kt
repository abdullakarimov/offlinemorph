package com.offlinemorph.android.feature.flags

/**
 * Build-time feature flags that gate in-development feature families.
 *
 * Set a flag to `true` locally to enable a feature that is under active development.
 * All flags default to `false` until the feature reaches beta-ready status and passes
 * the Definition of Done in IMPLEMENTATION_PLAN.md.
 */
object FeatureFlags {
    /** Phase C — AI Aging and De-aging synthesis (FRAN re-aging model, aging feature pack). */
    val agingEnabled: Boolean = true

    /** Phase D — Hair and Makeup studio. */
    val hairMakeupEnabled: Boolean = true

    /** Phase E — Beautifying and virtual plastic surgery controls. */
    val beautifyEnabled: Boolean = true

    /** Phase F — Cinematic 3D relighting. */
    val relightEnabled: Boolean = false

    /** Phase G — Background matting and scene replacement. */
    val backgroundSwapEnabled: Boolean = false

    /**
     * Run NPU-verified models on the Qualcomm Hexagon NPU (QNN HTP) when available; see
     * [com.offlinemorph.android.core.ml.NpuPolicy]. Set to false to force CPU everywhere.
     */
    val npuAccelerationEnabled: Boolean = true
}
