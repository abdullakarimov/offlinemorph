package com.offlinemorph.android.core.ml

/**
 * Typed error envelope returned by all feature engines.
 *
 * Each engine maps its internal failures to one of these variants so callers
 * can handle errors uniformly without inspecting raw exceptions.
 */
sealed class EngineError {
    /** A required model file was not found in the models directory. */
    data class ModelNotFound(val modelName: String) : EngineError()
    /** The ONNX inference session threw an exception. */
    data class InferenceFailure(val cause: Throwable) : EngineError()
    /** Face detection found no usable face at the indicated pipeline stage. */
    data class NoFaceDetected(val stage: String) : EngineError()
    /** The input was structurally invalid before inference started. */
    data class InvalidInput(val reason: String) : EngineError()
}

/**
 * Common result wrapper used by all feature engines.
 *
 * Callers pattern-match on [Success] vs [Failure] rather than catching exceptions.
 */
sealed class EngineResult<out T> {
    data class Success<T>(val value: T) : EngineResult<T>()
    data class Failure(
        val error: EngineError,
        val statusMessage: String,
    ) : EngineResult<Nothing>()
}

/**
 * Uniform progress event emitted via the `onProgress` callback during engine execution.
 *
 * [fraction] is a value in [0f, 1f] where ≥ 0 means the fraction is known, or -1f when
 * indeterminate.
 */
data class EngineProgress(val message: String, val fraction: Float = -1f)
