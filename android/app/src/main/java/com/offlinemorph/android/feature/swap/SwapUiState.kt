package com.offlinemorph.android.feature.swap

import android.graphics.Bitmap
import android.net.Uri
import com.offlinemorph.android.feature.models.ModelSpec

/**
 * Swap pipeline execution state. Drives the swap progress card in the UI.
 *
 * - [Idle]    : no swap running; user has not yet initiated one (or result was cleared).
 * - [Loading] : a swap is in progress; [message] names the current pipeline phase.
 * - [Success] : swap completed; [resultBitmap] is the composited output ready for display.
 * - [Error]   : swap failed; [exception] carries the root cause.
 */
sealed interface SwapUiState {
    data object Idle : SwapUiState
    data class Loading(val message: String) : SwapUiState
    data class Success(val resultBitmap: Bitmap) : SwapUiState
    data class Error(val exception: Throwable) : SwapUiState
}

/**
 * Live state of a single model file in the Setup download list.
 */
sealed interface ModelItemState {
    /** Not yet attempted this session. */
    data object Idle : ModelItemState
    /** File already present on disk and valid. */
    data object Installed : ModelItemState
    /** Currently downloading; [fraction] is 0..1, or -1 when size is unknown. */
    data class Downloading(val fraction: Float) : ModelItemState
    /** Download finished this session. */
    data object Done : ModelItemState
    /** Download failed; [reason] carries a short error description. */
    data class Failed(val reason: String) : ModelItemState
}

/**
 * One row in the Setup model list — pairs a [ModelSpec] with its current [state].
 */
data class ModelDownloadItem(
    val spec: ModelSpec,
    val state: ModelItemState = ModelItemState.Idle,
)

/**
 * Full-screen state for the swap screen.
 *
 * [swapState] drives the swap pipeline card / progress dialog.
 * [isWorking] covers AI file import / download operations (separate from swap).
 */
data class SwapScreenState(
    val sourceUri: Uri? = null,
    val targetUri: Uri? = null,
    /** True only during AI file import / download operations. See [swapState] for swap progress. */
    val isWorking: Boolean = false,
    val deviceCapabilityTitle: String = "Device capability not assessed.",
    val deviceCapabilityMessage: String = "Tap refresh or restart app to evaluate local device profile.",
    val modelStatusMessage: String = "AI pack validation not started.",
    val modelInstallMessage: String = "Import or download required AI files into app storage.",
    val modelDirectoryPath: String = "",
    val isAiPackReady: Boolean = false,
    val showMissingAiPackAlert: Boolean = false,
    val swapState: SwapUiState = SwapUiState.Idle,
    /** When true, the enhancer model runs after the inswapper to sharpen and upscale the swapped face. */
    val isEnhancerEnabled: Boolean = false,
    /** True when GFPGANv1.4.onnx is present in the models directory. */
    val isEnhancerModelAvailable: Boolean = false,
    /** Supplemental text shown below the result preview (e.g. save path after export). */
    val outputBitmapInfo: String? = null,
    /** Thumbnail bitmaps for every face detected in the current target image (sorted best-first). */
    val detectedTargetFaces: List<android.graphics.Bitmap> = emptyList(),
    /** Index of the face the user wants to replace when [faceFilterMode] is SPECIFIC. */
    val selectedTargetFaceIndex: Int = 0,
    /** True while target-face detection is running after the user picks a target image. */
    val isDetectingFaces: Boolean = false,
    /** Which faces in the target to swap. */
    val faceFilterMode: com.offlinemorph.android.core.ml.FaceFilterMode = com.offlinemorph.android.core.ml.FaceFilterMode.SPECIFIC,
    /** True when genderage.onnx is installed — enables Male/Female filter modes. */
    val isGenderModelAvailable: Boolean = false,
    /** Per-model download rows shown in the Setup screen. Populated by [refreshModelStatus]. */
    val downloadItems: List<ModelDownloadItem> = emptyList(),
) {
    val canSwap: Boolean
        get() = sourceUri != null && targetUri != null && !isWorking && !isDetectingFaces && swapState !is SwapUiState.Loading
}

