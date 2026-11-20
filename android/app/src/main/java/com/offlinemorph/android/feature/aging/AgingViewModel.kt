package com.offlinemorph.android.feature.aging

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.offlinemorph.android.core.image.AndroidBitmapLoader
import com.offlinemorph.android.core.ml.EngineError
import com.offlinemorph.android.core.ml.EngineResult
import com.offlinemorph.android.core.ml.aging.AgingEngine
import com.offlinemorph.android.core.ml.aging.AgingRequest
import com.offlinemorph.android.core.ml.OnnxFaceAnalyzer
import com.offlinemorph.android.core.ml.OrtSessionFactory
import com.offlinemorph.android.core.ml.aging.OnDeviceAgingEngine
import com.offlinemorph.android.feature.flags.FeatureFlags
import com.offlinemorph.android.feature.models.ModelPaths
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class AgingViewModel(
    application: Application,
) : AndroidViewModel(application) {

    // Engines are class-body properties: AndroidViewModelFactory needs a plain (Application) constructor.
    private val agingEngine: AgingEngine = ModelPaths.appModelsDirectory(application).let { dir ->
        OnDeviceAgingEngine(
            modelsDirectory = dir,
            faceAnalyzer = OnnxFaceAnalyzer(dir, OrtSessionFactory()),
        )
    }
    private val bitmapLoader = AndroidBitmapLoader(application.contentResolver)

    private val _uiState = MutableStateFlow(
        AgingScreenState(
            agingState = if (FeatureFlags.agingEnabled) AgingUiState.Idle else AgingUiState.Unavailable,
        ),
    )
    val uiState: StateFlow<AgingScreenState> = _uiState.asStateFlow()

    fun onSourceSelected(uri: Uri?) {
        _uiState.update { it.copy(sourceUri = uri, agingState = AgingUiState.Idle) }
    }

    fun onAgeOffsetChanged(years: Int) {
        _uiState.update { it.copy(ageOffsetYears = years.coerceIn(-50, 50)) }
    }

    fun onIntensityChanged(intensity: Float) {
        _uiState.update { it.copy(intensity = intensity.coerceIn(0f, 1f)) }
    }

    fun clearResult() {
        _uiState.update { it.copy(agingState = AgingUiState.Idle) }
    }

    fun runAging() {
        if (!FeatureFlags.agingEnabled) return
        val sourceUri = _uiState.value.sourceUri ?: return

        viewModelScope.launch {
            _uiState.update { it.copy(agingState = AgingUiState.Loading("Loading source image…"), isWorking = true) }

            val bitmapResult = withContext(Dispatchers.IO) { runCatching { bitmapLoader.load(sourceUri, maxDimensionPx = MAX_IMAGE_PX) } }
            if (bitmapResult.isFailure) {
                _uiState.update {
                    it.copy(
                        agingState = AgingUiState.Error("Failed to decode source image."),
                        isWorking = false,
                    )
                }
                return@launch
            }

            val request = AgingRequest(
                sourceBitmap = bitmapResult.getOrThrow().bitmap,
                ageOffsetYears = _uiState.value.ageOffsetYears,
                intensity = _uiState.value.intensity,
            )

            val engineResult = withContext(Dispatchers.Default) {
                agingEngine.runAging(request) { progress ->
                    _uiState.update { it.copy(agingState = AgingUiState.Loading(progress.message)) }
                }
            }

            _uiState.update {
                it.copy(
                    isWorking = false,
                    agingState = when (engineResult) {
                        is EngineResult.Success -> AgingUiState.Success(engineResult.value.outputBitmap)
                        is EngineResult.Failure -> when (engineResult.error) {
                            is EngineError.ModelNotFound ->
                                AgingUiState.ModelNotReady
                            else -> AgingUiState.Error(engineResult.statusMessage)
                        }
                    },
                )
            }
        }
    }

    private companion object {
        /** Longest edge the source photo is decoded at; FRAN works on a 1024 px face crop. */
        const val MAX_IMAGE_PX = 3072
    }
}
