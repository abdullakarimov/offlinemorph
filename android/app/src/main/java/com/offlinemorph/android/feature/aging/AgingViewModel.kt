package com.offlinemorph.android.feature.aging

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.offlinemorph.android.core.image.AndroidBitmapLoader
import com.offlinemorph.android.core.ml.EngineResult
import com.offlinemorph.android.core.ml.aging.AgingEngine
import com.offlinemorph.android.core.ml.aging.AgingRequest
import com.offlinemorph.android.core.ml.aging.StubAgingEngine
import com.offlinemorph.android.feature.flags.FeatureFlags
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

    private val agingEngine: AgingEngine = StubAgingEngine()
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

            val bitmapResult = runCatching { bitmapLoader.load(sourceUri) }
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
                            is com.offlinemorph.android.core.ml.EngineError.ModelNotFound ->
                                AgingUiState.ModelNotReady
                            else -> AgingUiState.Error(engineResult.statusMessage)
                        }
                    },
                )
            }
        }
    }
}
