package com.offlinemorph.android.feature.models

/**
 * Classifies a model's install requirement and feature relationship.
 *
 * - [CORE]         Required for the base app to function.
 * - [OPTIONAL]     Enhances an existing feature; safe to skip.
 * - [FEATURE_PACK] Belongs to a specific optional feature family; downloaded on demand.
 */
enum class ModelTier { CORE, OPTIONAL, FEATURE_PACK }

data class ModelSpec(
    val fileName: String,
    val role: String,
    val downloadUrls: List<String> = emptyList(),
    /** Legacy required flag; prefer [tier] for new code. */
    val required: Boolean = true,
    val tier: ModelTier = if (required) ModelTier.CORE else ModelTier.OPTIONAL,
    /** Semantic version of the model artifact (e.g. "1.0.0"). Empty string = unversioned. */
    val version: String = "",
    /** Expected SHA-256 hex digest for integrity validation. Null = no checksum enforced. */
    val sha256: String? = null,
    /** Feature pack identifier this model belongs to (e.g. "aging", "hair_makeup"). Empty = base pack. */
    val featurePack: String = "",
)

object ModelCatalog {
    const val DETECTOR     = "det_10g.onnx"
    const val LANDMARKS_2D = "2d106det.onnx"
    const val LANDMARKS_3D = "1k3d68.onnx"
    const val RECOGNIZER   = "w600k_r50.onnx"
    const val INSWAPPER    = "inswapper_128.onnx"
    const val GFPGAN       = "GFPGANv1.4.onnx"
    const val GENDERAGE    = "genderage.onnx"

    val requiredModels: List<ModelSpec> = listOf(
        ModelSpec(
            fileName = DETECTOR,
            role = "face detection",
            downloadUrls = listOf(
                "https://huggingface.co/halllooo/buffalo_l/resolve/main/det_10g.onnx",
            ),
        ),
        ModelSpec(
            fileName = LANDMARKS_2D,
            role = "2D facial landmarks",
            downloadUrls = listOf(
                "https://huggingface.co/halllooo/buffalo_l/resolve/main/2d106det.onnx",
            ),
        ),
        ModelSpec(
            fileName = LANDMARKS_3D,
            role = "3D facial landmarks",
            downloadUrls = listOf(
                "https://huggingface.co/halllooo/buffalo_l/resolve/main/1k3d68.onnx",
            ),
        ),
        ModelSpec(
            fileName = RECOGNIZER,
            role = "recognition embedding",
            downloadUrls = listOf(
                "https://huggingface.co/halllooo/buffalo_l/resolve/main/w600k_r50.onnx",
            ),
        ),
        ModelSpec(
            fileName = INSWAPPER,
            role = "face swap inference",
            downloadUrls = listOf(
                "https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx",
            ),
        ),
    )

    val optionalModels: List<ModelSpec> = listOf(
        ModelSpec(
            fileName = GFPGAN,
            role = "AI face enhancement (GFPGAN v1.4)",
            required = false,
            downloadUrls = listOf(
                "https://huggingface.co/countfloyd/deepfake/resolve/main/GFPGANv1.4.onnx",
            ),
        ),
        ModelSpec(
            fileName = GENDERAGE,
            role = "gender & age classification (for Male/Female face filter)",
            required = false,
            downloadUrls = listOf(
                "https://huggingface.co/halllooo/buffalo_l/resolve/main/genderage.onnx",
            ),
        ),
    )

    val allModels: List<ModelSpec> get() = requiredModels + optionalModels

    fun expectedFileNames(): List<String> = requiredModels.map { it.fileName }
}
