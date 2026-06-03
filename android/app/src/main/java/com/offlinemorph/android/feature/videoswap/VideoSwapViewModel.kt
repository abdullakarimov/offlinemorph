package com.offlinemorph.android.feature.videoswap

import android.app.Application
import android.content.ContentValues
import android.content.Intent
import android.net.Uri
import android.os.Environment
import android.provider.MediaStore
import androidx.core.content.FileProvider
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.offlinemorph.android.core.image.AndroidBitmapLoader
import com.offlinemorph.android.core.image.BitmapLoader
import com.offlinemorph.android.core.ml.FaceFilterMode
import com.offlinemorph.android.core.ml.OnDeviceFaceSwapEngine
import com.offlinemorph.android.core.ml.OnnxFaceAnalyzer
import com.offlinemorph.android.core.ml.OrtSessionFactory
import com.offlinemorph.android.core.video.OnDeviceVideoSwapEngine
import com.offlinemorph.android.core.video.VideoSwapEngine
import com.offlinemorph.android.core.video.VideoSwapRequest
import com.offlinemorph.android.feature.models.ModelCatalog
import com.offlinemorph.android.feature.models.ModelPaths
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.io.File

class VideoSwapViewModel(
    application: Application,
    private val videoSwapEngine: VideoSwapEngine,
) : AndroidViewModel(application) {

    constructor(application: Application) : this(
        application = application,
        videoSwapEngine = OnDeviceVideoSwapEngine(
            context = application,
            faceSwapEngine = OnDeviceFaceSwapEngine(
                modelsDirectory = ModelPaths.appModelsDirectory(application),
                faceAnalyzer = OnnxFaceAnalyzer(
                    modelsDirectory = ModelPaths.appModelsDirectory(application),
                    sessionFactory = OrtSessionFactory(),
                ),
                sessionFactory = OrtSessionFactory(),
            ),
            outputDir = File(application.filesDir, "video_swap_output"),
        ),
    )

    private val bitmapLoader: BitmapLoader = AndroidBitmapLoader(application.contentResolver)
    private val _uiState = MutableStateFlow(VideoSwapScreenState())
    val uiState: StateFlow<VideoSwapScreenState> = _uiState.asStateFlow()

    private var swapJob: Job? = null

    init {
        refreshModelAvailability()
    }

    fun onSourceSelected(uri: Uri) {
        viewModelScope.launch {
            val bitmap = bitmapLoader.load(uri).bitmap
            _uiState.update { it.copy(sourceUri = uri, sourceBitmap = bitmap, swapState = VideoSwapUiState.Idle) }
        }
    }

    fun onVideoSelected(uri: Uri) {
        _uiState.update { it.copy(targetVideoUri = uri, swapState = VideoSwapUiState.Idle) }
    }

    fun setEnhancerEnabled(enabled: Boolean) {
        _uiState.update { it.copy(enhancerEnabled = enabled) }
    }

    fun setFaceFilterMode(mode: FaceFilterMode) {
        _uiState.update { it.copy(faceFilterMode = mode) }
    }

    /** Call after models are installed to refresh whether genderage.onnx is available. */
    fun refreshModelAvailability() {
        val modelsDir = ModelPaths.appModelsDirectory(getApplication())
        val genderAvailable = File(modelsDir, ModelCatalog.GENDERAGE).isFile
        _uiState.update { it.copy(isGenderModelAvailable = genderAvailable) }
    }

    fun startVideoSwap() {
        val state = _uiState.value
        val sourceBitmap = state.sourceBitmap ?: return
        val targetUri = state.targetVideoUri ?: return

        swapJob = viewModelScope.launch {
            _uiState.update { it.copy(swapState = VideoSwapUiState.Loading(statusMessage = "Starting…")) }

            val result = videoSwapEngine.swapVideo(
                request = VideoSwapRequest(
                    sourceBitmap = sourceBitmap,
                    targetVideoUri = targetUri,
                    enhancerEnabled = state.enhancerEnabled,
                    faceFilterMode  = state.faceFilterMode,
                ),
                cancellationCheck = { swapJob?.isCancelled == true },
                onProgress = { current, total, msg ->
                    _uiState.update { it.copy(swapState = VideoSwapUiState.Loading(current, total, msg)) }
                },
            )

            _uiState.update {
                it.copy(
                    swapState = if (result.outputFile != null)
                        VideoSwapUiState.Success(result.outputFile)
                    else
                        VideoSwapUiState.Error(result.statusMessage),
                )
            }
        }
    }

    fun cancelVideoSwap() {
        swapJob?.cancel()
        _uiState.update { it.copy(swapState = VideoSwapUiState.Idle) }
    }

    fun shareOutput() {
        val file = (_uiState.value.swapState as? VideoSwapUiState.Success)?.outputFile ?: return
        val contentUri = FileProvider.getUriForFile(
            getApplication(),
            "${getApplication<Application>().packageName}.provider",
            file,
        )
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "video/mp4"
            putExtra(Intent.EXTRA_STREAM, contentUri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        getApplication<Application>().startActivity(Intent.createChooser(intent, "Share video").apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        })
    }

    fun saveOutput() {
        val file = (_uiState.value.swapState as? VideoSwapUiState.Success)?.outputFile ?: return
        viewModelScope.launch {
            val resolver = getApplication<Application>().contentResolver
            val values = ContentValues().apply {
                put(MediaStore.Video.Media.DISPLAY_NAME, file.name)
                put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
                put(MediaStore.Video.Media.RELATIVE_PATH, "${Environment.DIRECTORY_MOVIES}/OfflineMorph")
            }
            val uri = resolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values)
            uri?.let {
                resolver.openOutputStream(it)?.use { out ->
                    file.inputStream().use { inp -> inp.copyTo(out) }
                }
            }
        }
    }
}
