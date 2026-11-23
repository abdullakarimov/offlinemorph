package com.offlinemorph.android.feature.hairmakeup

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.offlinemorph.android.core.image.AndroidBitmapLoader
import com.offlinemorph.android.core.ml.EngineError
import com.offlinemorph.android.core.ml.EngineResult
import com.offlinemorph.android.core.ml.OnnxFaceAnalyzer
import com.offlinemorph.android.core.ml.OrtSessionFactory
import com.offlinemorph.android.core.ml.hairmakeup.FacialHairStyle
import com.offlinemorph.android.core.ml.hairmakeup.HairStyle
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

    // Engines are class-body properties: AndroidViewModelFactory needs a plain (Application) constructor.
    private val hairMakeupEngine: HairMakeupEngine = ModelPaths.appModelsDirectory(application).let { dir ->
        val sessions = OrtSessionFactory()
        OnDeviceHairMakeupEngine(modelsDirectory = dir, faceAnalyzer = OnnxFaceAnalyzer(dir, sessions), sessionFactory = sessions)
    }
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

    fun onHairStyleSelected(style: HairStyle) {
        _uiState.update { it.copy(hairStyle = style) }
    }

    fun onHairSecondaryColorSelected(hex: String?) {
        _uiState.update { it.copy(hairSecondaryColorHex = hex) }
    }

    fun onEyeshadowColorSelected(hex: String?) {
        _uiState.update { it.copy(eyeshadowColorHex = hex) }
    }

    fun onEyelinerColorSelected(hex: String?) {
        _uiState.update { it.copy(eyelinerColorHex = hex) }
    }

    fun onBlushColorSelected(hex: String?) {
        _uiState.update { it.copy(blushColorHex = hex) }
    }

    fun onFacialHairSelected(style: FacialHairStyle) {
        _uiState.update { it.copy(facialHair = style) }
    }

    fun onFacialHairColorSelected(hex: String?) {
        _uiState.update { it.copy(facialHairColorHex = hex) }
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

            val bitmapResult = withContext(Dispatchers.IO) { runCatching { bitmapLoader.load(sourceUri, maxDimensionPx = 4096) } }
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
                hairStyle = state.hairStyle,
                hairSecondaryColorHex = state.hairSecondaryColorHex,
                lipColorHex = state.lipColorHex,
                eyeshadowColorHex = state.eyeshadowColorHex,
                eyelinerColorHex = state.eyelinerColorHex,
                blushColorHex = state.blushColorHex,
                facialHair = state.facialHair,
                facialHairColorHex = state.facialHairColorHex,
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
