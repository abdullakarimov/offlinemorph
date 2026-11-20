package com.offlinemorph.android.feature.ancestry

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.offlinemorph.android.core.image.AndroidBitmapLoader
import com.offlinemorph.android.core.ml.EngineError
import com.offlinemorph.android.core.ml.EngineResult
import com.offlinemorph.android.core.ml.OnDeviceFaceSwapEngine
import com.offlinemorph.android.core.ml.OnnxFaceAnalyzer
import com.offlinemorph.android.core.ml.OrtSessionFactory
import com.offlinemorph.android.core.ml.ancestry.AncestryEngine
import com.offlinemorph.android.core.ml.ancestry.AncestryRequest
import com.offlinemorph.android.core.ml.ancestry.OnDeviceAncestryEngine
import com.offlinemorph.android.feature.flags.FeatureFlags
import com.offlinemorph.android.feature.models.ModelPaths
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class AncestryViewModel(
    application: Application,
) : AndroidViewModel(application) {

    // Engines are class-body properties: AndroidViewModelFactory needs a plain (Application) constructor.
    private val ancestryEngine: AncestryEngine = ModelPaths.appModelsDirectory(application).let { dir ->
        val sessions = OrtSessionFactory()
        val analyzer = OnnxFaceAnalyzer(dir, sessions)
        OnDeviceAncestryEngine(
            modelsDirectory = dir,
            faceAnalyzer = analyzer,
            swapEngine = OnDeviceFaceSwapEngine(dir, analyzer, sessions),
        )
    }
    private val bitmapLoader = AndroidBitmapLoader(application.contentResolver)

    private val _uiState = MutableStateFlow(
        AncestryScreenState(
            ancestryState = if (FeatureFlags.ancestryEnabled) AncestryUiState.Idle else AncestryUiState.Unavailable,
        ),
    )
    val uiState: StateFlow<AncestryScreenState> = _uiState.asStateFlow()

    fun onFaceASelected(uri: Uri?) = updateInputs { it.copy(faceAUri = uri ?: it.faceAUri) }

    fun onFaceBSelected(uri: Uri?) = updateInputs { it.copy(faceBUri = uri ?: it.faceBUri) }

    fun onBaseSelected(uri: Uri?) = updateInputs { it.copy(baseUri = uri ?: it.baseUri) }

    fun clearBase() = updateInputs { it.copy(baseUri = null) }

    fun onBlendRatioChanged(ratio: Float) {
        _uiState.update { it.copy(blendRatio = ratio.coerceIn(0f, 1f)) }
    }

    fun onIntensityChanged(intensity: Float) {
        _uiState.update { it.copy(intensity = intensity.coerceIn(0f, 1f)) }
    }

    fun clearResult() {
        _uiState.update { it.copy(ancestryState = AncestryUiState.Idle) }
    }

    private fun updateInputs(change: (AncestryScreenState) -> AncestryScreenState) {
        _uiState.update { change(it).copy(ancestryState = AncestryUiState.Idle) }
    }

    fun runAncestry() {
        if (!FeatureFlags.ancestryEnabled) return
        val state = _uiState.value
        if (!state.canRun) return
        val faceAUri = state.faceAUri ?: return
        val faceBUri = state.faceBUri ?: return
        val baseUri = state.baseUri ?: faceAUri

        viewModelScope.launch {
            _uiState.update { it.copy(ancestryState = AncestryUiState.Loading("Loading photos…"), isWorking = true) }

            val bitmaps = withContext(Dispatchers.IO) {
                runCatching {
                    Triple(
                        bitmapLoader.load(faceAUri, MAX_IMAGE_PX).bitmap,
                        bitmapLoader.load(faceBUri, MAX_IMAGE_PX).bitmap,
                        bitmapLoader.load(baseUri, MAX_IMAGE_PX).bitmap,
                    )
                }
            }
            val (faceA, faceB, base) = bitmaps.getOrElse {
                _uiState.update {
                    it.copy(ancestryState = AncestryUiState.Error("Failed to decode one of the photos."), isWorking = false)
                }
                return@launch
            }

            val request = AncestryRequest(
                targetBitmap = base,
                faceABitmap = faceA,
                faceBBitmap = faceB,
                blendRatio = state.blendRatio,
                intensity = state.intensity,
            )
            val engineResult = withContext(Dispatchers.Default) {
                ancestryEngine.runAncestry(request) { progress ->
                    _uiState.update { it.copy(ancestryState = AncestryUiState.Loading(progress.message)) }
                }
            }

            _uiState.update {
                it.copy(
                    isWorking = false,
                    ancestryState = when (engineResult) {
                        is EngineResult.Success ->
                            AncestryUiState.Success(engineResult.value.outputBitmap, engineResult.value.appliedRatio)
                        is EngineResult.Failure -> when (engineResult.error) {
                            is EngineError.ModelNotFound -> AncestryUiState.ModelNotReady
                            else -> AncestryUiState.Error(engineResult.statusMessage)
                        }
                    },
                )
            }
        }
    }

    private companion object {
        /** Longest edge photos are decoded at; the identity pipeline never needs more. */
        const val MAX_IMAGE_PX = 2048
    }
}
