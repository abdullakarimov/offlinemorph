package com.offlinemorph.android.feature.beautify

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.offlinemorph.android.core.image.AndroidBitmapLoader
import com.offlinemorph.android.core.ml.EngineResult
import com.offlinemorph.android.core.ml.beautify.BeautifyEngine
import com.offlinemorph.android.core.ml.beautify.BeautifyRequest
import com.offlinemorph.android.core.ml.beautify.OnDeviceBeautifyEngine
import com.offlinemorph.android.feature.flags.FeatureFlags
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class BeautifyViewModel(application: Application) : AndroidViewModel(application) {

    private val beautifyEngine: BeautifyEngine = OnDeviceBeautifyEngine()
    private val bitmapLoader = AndroidBitmapLoader(application.contentResolver)

    private val _uiState = MutableStateFlow(
        BeautifyScreenState(
            beautifyState = if (FeatureFlags.beautifyEnabled) BeautifyUiState.Idle else BeautifyUiState.Unavailable,
        ),
    )
    val uiState: StateFlow<BeautifyScreenState> = _uiState.asStateFlow()

    fun onSourceSelected(uri: Uri?) {
        _uiState.update { it.copy(sourceUri = uri, beautifyState = BeautifyUiState.Idle) }
    }

    fun onSkinSmoothingChanged(v: Float) {
        _uiState.update { it.copy(skinSmoothing = v.coerceIn(0f, 1f)) }
    }

    fun onEyeEnlargeChanged(v: Float) {
        _uiState.update { it.copy(eyeEnlarge = v.coerceIn(0f, 1f)) }
    }

    fun onFaceSlimChanged(v: Float) {
        _uiState.update { it.copy(faceSlim = v.coerceIn(0f, 1f)) }
    }

    fun onTeethWhitenChanged(v: Float) {
        _uiState.update { it.copy(teethWhiten = v.coerceIn(0f, 1f)) }
    }

    fun clearResult() {
        _uiState.update { it.copy(beautifyState = BeautifyUiState.Idle) }
    }

    fun runBeautify() {
        if (!FeatureFlags.beautifyEnabled) return
        val sourceUri = _uiState.value.sourceUri ?: return

        viewModelScope.launch {
            _uiState.update { it.copy(beautifyState = BeautifyUiState.Loading("Loading image…"), isWorking = true) }

            val bitmapResult = runCatching { bitmapLoader.load(sourceUri) }
            if (bitmapResult.isFailure) {
                _uiState.update {
                    it.copy(beautifyState = BeautifyUiState.Error("Failed to load image."), isWorking = false)
                }
                return@launch
            }

            val state = _uiState.value
            val request = BeautifyRequest(
                sourceBitmap = bitmapResult.getOrThrow().bitmap,
                skinSmoothing = state.skinSmoothing,
                eyeEnlarge = state.eyeEnlarge,
                faceSlim = state.faceSlim,
                teethWhiten = state.teethWhiten,
            )

            val engineResult = withContext(Dispatchers.Default) {
                beautifyEngine.runBeautify(request) { progress ->
                    _uiState.update { it.copy(beautifyState = BeautifyUiState.Loading(progress.message)) }
                }
            }

            _uiState.update {
                it.copy(
                    isWorking = false,
                    beautifyState = when (engineResult) {
                        is EngineResult.Success -> BeautifyUiState.Success(engineResult.value.outputBitmap)
                        is EngineResult.Failure -> BeautifyUiState.Error(engineResult.statusMessage)
                    },
                )
            }
        }
    }
}
