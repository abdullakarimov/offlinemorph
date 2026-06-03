package com.offlinemorph.android.feature.hairmakeup

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.offlinemorph.android.core.image.AndroidBitmapLoader
import com.offlinemorph.android.core.ml.EngineError
import com.offlinemorph.android.core.ml.EngineResult
import com.offlinemorph.android.core.ml.hairmakeup.HairMakeupEngine
import com.offlinemorph.android.core.ml.hairmakeup.HairMakeupRequest
import com.offlinemorph.android.core.ml.hairmakeup.OnDeviceHairMakeupEngine
import com.offlinemorph.android.feature.flags.FeatureFlags
import com.offlinemorph.android.feature.models.ModelPaths
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class HairMakeupViewModel(application: Application) : AndroidViewModel(application) {

    private val hairMakeupEngine: HairMakeupEngine =
        OnDeviceHairMakeupEngine(modelsDirectory = ModelPaths.appModelsDirectory(application))
    private val bitmapLoader = AndroidBitmapLoader(application.contentResolver)

    private val _uiState = MutableStateFlow(
        HairMakeupScreenState(
            hairMakeupState = if (FeatureFlags.hairMakeupEnabled) HairMakeupUiState.Idle else HairMakeupUiState.Unavailable,
        ),
    )
    val uiState: StateFlow<HairMakeupScreenState> = _uiState.asStateFlow()

    fun onSourceSelected(uri: Uri?) {
        _uiState.update { it.copy(sourceUri = uri, hairMakeupState = HairMakeupUiState.Idle) }
    }

    fun onHairColorSelected(hex: String?) {
        _uiState.update { it.copy(hairColorHex = hex) }
    }

    fun onLipColorSelected(hex: String?) {
        _uiState.update { it.copy(lipColorHex = hex) }
    }

    fun onIntensityChanged(v: Float) {
        _uiState.update { it.copy(intensity = v.coerceIn(0f, 1f)) }
    }

    fun clearResult() {
        _uiState.update { it.copy(hairMakeupState = HairMakeupUiState.Idle) }
    }

    fun runHairMakeup() {
        if (!FeatureFlags.hairMakeupEnabled) return
        val sourceUri = _uiState.value.sourceUri ?: return

        viewModelScope.launch {
            _uiState.update {
                it.copy(hairMakeupState = HairMakeupUiState.Loading("Loading image…"), isWorking = true)
            }

            val bitmapResult = runCatching { bitmapLoader.load(sourceUri) }
            if (bitmapResult.isFailure) {
                _uiState.update {
                    it.copy(hairMakeupState = HairMakeupUiState.Error("Failed to load image."), isWorking = false)
                }
                return@launch
            }

            val state = _uiState.value
            val request = HairMakeupRequest(
                sourceBitmap = bitmapResult.getOrThrow().bitmap,
                hairColorHex = state.hairColorHex,
                lipColorHex = state.lipColorHex,
                intensity = state.intensity,
            )

            val engineResult = withContext(Dispatchers.Default) {
                hairMakeupEngine.runHairMakeup(request) { progress ->
                    _uiState.update { it.copy(hairMakeupState = HairMakeupUiState.Loading(progress.message)) }
                }
            }

            _uiState.update {
                it.copy(
                    isWorking = false,
                    hairMakeupState = when (engineResult) {
                        is EngineResult.Success -> HairMakeupUiState.Success(engineResult.value.outputBitmap)
                        is EngineResult.Failure -> when (engineResult.error) {
                            is EngineError.ModelNotFound -> HairMakeupUiState.ModelNotReady
                            else -> HairMakeupUiState.Error(engineResult.statusMessage)
                        }
                    },
                )
            }
        }
    }
}
