package com.offlinemorph.android.feature.swap

import android.app.Application
import android.content.ContentValues
import android.net.Uri
import android.os.Environment
import android.provider.MediaStore
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.offlinemorph.android.core.image.AndroidBitmapLoader
import com.offlinemorph.android.core.image.BitmapLoader
import com.offlinemorph.android.core.ml.FaceAnalyzer
import com.offlinemorph.android.core.ml.FaceSwapEngine
import com.offlinemorph.android.core.ml.OnDeviceFaceSwapEngine
import com.offlinemorph.android.core.ml.OnnxFaceAnalyzer
import com.offlinemorph.android.core.ml.OrtSessionFactory
import com.offlinemorph.android.core.ml.SwapRequest
import com.offlinemorph.android.feature.device.DeviceCapabilityAssessor
import com.offlinemorph.android.feature.device.ExecutionPolicyManager
import com.offlinemorph.android.feature.models.AndroidModelDownloader
import com.offlinemorph.android.feature.models.AndroidModelInstaller
import com.offlinemorph.android.feature.models.ModelCatalog
import com.offlinemorph.android.feature.models.ModelInstaller
import com.offlinemorph.android.feature.models.LocalModelManager
import com.offlinemorph.android.feature.models.ModelManager
import com.offlinemorph.android.feature.models.ModelPaths
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class SwapViewModel(
    application: Application,
    private val modelManager: ModelManager,
    private val modelInstaller: ModelInstaller,
    private val modelDownloader: AndroidModelDownloader,
    private val faceSwapEngine: FaceSwapEngine,
    private val faceAnalyzer: FaceAnalyzer,
) : AndroidViewModel(application) {
    constructor(application: Application) : this(
        application = application,
        modelManager = LocalModelManager(ModelPaths.appModelsDirectory(application)),
        modelInstaller = AndroidModelInstaller(
            application.contentResolver,
            ModelPaths.appModelsDirectory(application),
        ),
        modelDownloader = AndroidModelDownloader(
            modelsDirectory = ModelPaths.appModelsDirectory(application),
        ),
        faceAnalyzer = OnnxFaceAnalyzer(
            modelsDirectory = ModelPaths.appModelsDirectory(application),
            sessionFactory = OrtSessionFactory(),
        ),
        faceSwapEngine = OnDeviceFaceSwapEngine(
            modelsDirectory = ModelPaths.appModelsDirectory(application),
            faceAnalyzer = OnnxFaceAnalyzer(
                modelsDirectory = ModelPaths.appModelsDirectory(application),
                sessionFactory = OrtSessionFactory(),
            ),
            sessionFactory = OrtSessionFactory(),
        ),
    )

    private val bitmapLoader: BitmapLoader = AndroidBitmapLoader(application.contentResolver)
    private val capabilityAssessor = DeviceCapabilityAssessor(application)
    private val policyManager = ExecutionPolicyManager(application)
    private val _uiState = MutableStateFlow(SwapScreenState())
    val uiState: StateFlow<SwapScreenState> = _uiState.asStateFlow()

    init {
        refreshDeviceCapability()
        refreshModelStatus()
    }

    fun refreshDeviceCapability() {
        val capability = capabilityAssessor.assess()
        _uiState.update {
            it.copy(
                deviceCapabilityTitle = "Device Tier: ${capability.tier}",
                deviceCapabilityMessage = "${capability.summary}\n${capability.recommendation}",
            )
        }
    }

    fun onSourceSelected(uri: Uri?) {
        _uiState.update {
            it.copy(sourceUri = uri, swapState = SwapUiState.Idle, outputBitmapInfo = null)
        }
    }

    fun onTargetSelected(uri: Uri?) {
        _uiState.update {
            it.copy(
                targetUri = uri,
                swapState = SwapUiState.Idle,
                outputBitmapInfo = null,
                detectedTargetFaces = emptyList(),
                selectedTargetFaceIndex = 0,
            )
        }
        if (uri != null) {
            detectTargetFaces(uri)
        }
    }

    fun selectTargetFace(index: Int) {
        _uiState.update { it.copy(selectedTargetFaceIndex = index) }
    }

    private fun detectTargetFaces(uri: Uri) {
        viewModelScope.launch {
            _uiState.update { it.copy(isDetectingFaces = true) }
            val bitmapResult = runCatching { bitmapLoader.load(uri) }
            if (bitmapResult.isFailure) {
                _uiState.update { it.copy(isDetectingFaces = false) }
                return@launch
            }
            val bitmap = bitmapResult.getOrThrow().bitmap
            val analysis = withContext(Dispatchers.Default) {
                faceAnalyzer.analyze(bitmap)
            }
            val thumbnails = analysis.allDetectedFaces.mapNotNull { it.thumbnail }
            _uiState.update {
                it.copy(
                    isDetectingFaces = false,
                    detectedTargetFaces = thumbnails,
                    selectedTargetFaceIndex = 0,
                )
            }
        }
    }

    fun toggleEnhancer(enabled: Boolean) {
        _uiState.update { it.copy(isEnhancerEnabled = enabled) }
    }

    fun setFaceFilterMode(mode: com.offlinemorph.android.core.ml.FaceFilterMode) {
        _uiState.update { it.copy(faceFilterMode = mode) }
    }

    fun refreshModelStatus() {
        viewModelScope.launch {
            val status = modelManager.getStatus()
            val message = if (status.ready) {
                "AI pack is ready for local inference. Installed: ${status.installedModels.joinToString { it.fileName }}"
            } else {
                "Installed: ${status.installedModels.joinToString { it.fileName }.ifBlank { "none" }}. Missing: ${status.missingModels.joinToString { it.fileName }}"
            }
            val hasAnyAiFiles = status.installedModels.isNotEmpty() || status.optionalInstalledModels.isNotEmpty()
            val shouldPromptInstall = !status.ready && !hasAnyAiFiles
            val enhancerAvailable = status.optionalInstalledModels.any { it.fileName == ModelCatalog.GFPGAN }
            val genderModelAvailable = status.optionalInstalledModels.any { it.fileName == ModelCatalog.GENDERAGE }
            val installHint = buildString {
                append("Required AI files: ${ModelCatalog.expectedFileNames().joinToString()}")
                if (status.optionalMissingModels.isNotEmpty()) {
                    append(". Optional (not installed): ${status.optionalMissingModels.joinToString { it.fileName }}")
                } else if (status.optionalInstalledModels.isNotEmpty()) {
                    append(". Optional: ${status.optionalInstalledModels.joinToString { it.fileName }} ✓")
                }
            }

            // Build per-model download rows for the Setup screen.
            val installedNames = (status.installedModels + status.optionalInstalledModels).map { it.fileName }.toSet()
            val items = ModelCatalog.allModels.map { spec ->
                ModelDownloadItem(
                    spec = spec,
                    state = if (spec.fileName in installedNames) ModelItemState.Installed else ModelItemState.Idle,
                )
            }

            _uiState.update { current ->
                current.copy(
                    modelStatusMessage = message,
                    modelDirectoryPath = status.modelDirectoryPath,
                    modelInstallMessage = installHint,
                    isAiPackReady = status.ready,
                    showMissingAiPackAlert = if (status.ready || hasAnyAiFiles) {
                        false
                    } else {
                        current.showMissingAiPackAlert || shouldPromptInstall
                    },
                    isEnhancerModelAvailable = enhancerAvailable,
                    isGenderModelAvailable = genderModelAvailable,
                    downloadItems = if (current.isWorking) current.downloadItems else items,
                )
            }
        }
    }

    fun dismissMissingAiPackAlert() {
        _uiState.update { it.copy(showMissingAiPackAlert = false) }
    }

    fun installModels(uris: List<Uri>) {
        if (uris.isEmpty()) {
            return
        }

        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    isWorking = true,
                    modelInstallMessage = "Importing ${uris.size} selected AI file(s)...",
                )
            }

            val result = modelInstaller.install(uris)
            val installMessage = buildString {
                append("Installed: ")
                append(result.installedModels.joinToString { it.fileName }.ifBlank { "none" })
                if (result.skippedFiles.isNotEmpty()) {
                    append(". Skipped: ")
                    append(result.skippedFiles.joinToString())
                }
                if (result.failedFiles.isNotEmpty()) {
                    append(". Failed: ")
                    append(result.failedFiles.joinToString())
                }
            }

            _uiState.update {
                it.copy(
                    isWorking = false,
                    modelInstallMessage = installMessage,
                )
            }
            refreshModelStatus()
        }
    }

    fun downloadModels() {
        runDownload(requiredOnly = true)
    }

    fun downloadAllModels() {
        runDownload(requiredOnly = false)
    }

    private fun runDownload(requiredOnly: Boolean) {
        viewModelScope.launch {
            _uiState.update { it.copy(isWorking = true, modelInstallMessage = "Starting download…") }

            val result = if (requiredOnly) {
                modelDownloader.downloadMissingRequiredModels(
                    onProgress = { msg -> _uiState.update { it.copy(modelInstallMessage = msg) } },
                    onFileProgress = { p -> updateItemProgress(p.fileName, p.fraction) },
                )
            } else {
                modelDownloader.downloadAllModels(
                    onProgress = { msg -> _uiState.update { it.copy(modelInstallMessage = msg) } },
                    onFileProgress = { p -> updateItemProgress(p.fileName, p.fraction) },
                )
            }

            val summary = buildString {
                append("Downloaded: ")
                append(result.downloadedModels.joinToString { it.fileName }.ifBlank { "none" })
                append(". Existing: ")
                append(result.skippedModels.joinToString { it.fileName }.ifBlank { "none" })
                if (result.failedModels.isNotEmpty()) {
                    append(". Failed: ")
                    append(result.failedModels.joinToString())
                }
            }

            _uiState.update { it.copy(isWorking = false, modelInstallMessage = summary) }
            refreshModelStatus()
        }
    }

    private fun updateItemProgress(fileName: String, fraction: Float) {
        _uiState.update { state ->
            val items = state.downloadItems.map { item ->
                if (item.spec.fileName != fileName) item
                else item.copy(
                    state = when {
                        fraction >= 1f -> ModelItemState.Done
                        fraction < 0f  -> ModelItemState.Downloading(-1f)
                        else           -> ModelItemState.Downloading(fraction)
                    },
                )
            }
            state.copy(downloadItems = items)
        }
    }

    fun startSwap() {
        if (!_uiState.value.isAiPackReady) {
            _uiState.update {
                it.copy(
                    showMissingAiPackAlert = true,
                    swapState = SwapUiState.Error(
                        IllegalStateException("AI files are missing. Open Setup to download or import required files first.")
                    ),
                )
            }
            return
        }

        val sourceUri = _uiState.value.sourceUri ?: return
        val targetUri = _uiState.value.targetUri ?: return

        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    swapState = SwapUiState.Loading("Decoding selected images..."),
                    outputBitmapInfo = null,
                )
            }

            // Decode bitmaps on the main thread (ContentResolver I/O), then hand off to Default.
            val sourceBitmapResult = runCatching { bitmapLoader.load(sourceUri) }
            val targetBitmapResult = runCatching { bitmapLoader.load(targetUri) }

            if (sourceBitmapResult.isFailure || targetBitmapResult.isFailure) {
                val cause = (sourceBitmapResult.exceptionOrNull() ?: targetBitmapResult.exceptionOrNull())
                    ?: Exception("Failed to decode one or both images.")
                _uiState.update { it.copy(swapState = SwapUiState.Error(cause)) }
                return@launch
            }

            val policy = policyManager.currentPolicy()
            val swapRequest = SwapRequest(
                sourceBitmap = sourceBitmapResult.getOrThrow().bitmap,
                targetBitmap = targetBitmapResult.getOrThrow().bitmap,
                // Enhancer is only permitted when both the user has enabled it AND the
                // current execution policy allows it (thermal / memory guardrail).
                enhancerEnabled = _uiState.value.isEnhancerEnabled && policy.enhancerEnabled,
                targetFaceIndex = _uiState.value.selectedTargetFaceIndex,
                faceFilterMode  = _uiState.value.faceFilterMode,
                maxImageSizePx  = policy.maxImageSizePx,
            )

            // Run the full inference pipeline off the main thread.
            // onProgress callbacks update _uiState directly — MutableStateFlow.update is thread-safe.
            val result = runCatching {
                withContext(Dispatchers.Default) {
                    faceSwapEngine.runSwap(swapRequest) { milestone ->
                        _uiState.update { it.copy(swapState = SwapUiState.Loading(milestone)) }
                    }
                }
            }

            _uiState.update { state ->
                result.fold(
                    onSuccess = { swapResult ->
                        val bitmap = swapResult.outputBitmap
                        if (bitmap != null) {
                            state.copy(
                                swapState = SwapUiState.Success(bitmap),
                                outputBitmapInfo = "Output: ${bitmap.width}×${bitmap.height}",
                            )
                        } else {
                            state.copy(
                                swapState = SwapUiState.Error(Exception(swapResult.statusMessage)),
                            )
                        }
                    },
                    onFailure = { error ->
                        state.copy(swapState = SwapUiState.Error(error))
                    },
                )
            }
        }
    }

    fun saveOutputImage() {
        val bitmap = (_uiState.value.swapState as? SwapUiState.Success)?.resultBitmap ?: return

        viewModelScope.launch {
            _uiState.update {
                it.copy(isWorking = true, outputBitmapInfo = "Saving output image...")
            }

            val app = getApplication<Application>()
            val resolver = app.contentResolver
            val fileName = "offlinemorph_swap_${System.currentTimeMillis()}.png"
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, fileName)
                put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                put(
                    MediaStore.Images.Media.RELATIVE_PATH,
                    Environment.DIRECTORY_PICTURES + "/OfflineMorph",
                )
            }

            val watermarked = withContext(Dispatchers.Default) {
                stampAiGeneratedLabel(bitmap)
            }

            val savedUri = runCatching {
                val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                    ?: error("Unable to create media store item")
                resolver.openOutputStream(uri).use { stream ->
                    requireNotNull(stream) { "Unable to open media store output stream" }
                    check(watermarked.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, stream)) {
                        "Bitmap compression failed"
                    }
                }
                uri
            }

            _uiState.update {
                if (savedUri.isSuccess) {
                    it.copy(
                        isWorking = false,
                        outputBitmapInfo = buildString {
                            append(it.outputBitmapInfo ?: "Output bitmap prepared")
                            append("\nSaved as: $fileName")
                        },
                    )
                } else {
                    it.copy(
                        isWorking = false,
                        outputBitmapInfo = "Save failed: ${savedUri.exceptionOrNull()?.message}",
                    )
                }
            }
        }
    }

    /**
     * Stamps a small "AI Generated" label into the bottom-right corner of [src].
     * Returns a new mutable copy — the original bitmap is not modified.
     */
    private fun stampAiGeneratedLabel(src: android.graphics.Bitmap): android.graphics.Bitmap {
        val out = src.copy(android.graphics.Bitmap.Config.ARGB_8888, true)
        val canvas = android.graphics.Canvas(out)
        val label = "AI Generated"
        val textSize = (out.width * 0.03f).coerceIn(18f, 48f)
        val padding = textSize * 0.6f

        val bgPaint = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
            color = android.graphics.Color.argb(140, 0, 0, 0)
            style = android.graphics.Paint.Style.FILL
        }
        val textPaint = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
            color = android.graphics.Color.WHITE
            this.textSize = textSize
            typeface = android.graphics.Typeface.create(android.graphics.Typeface.DEFAULT, android.graphics.Typeface.BOLD)
        }

        val textWidth = textPaint.measureText(label)
        val metrics = textPaint.fontMetrics
        val textHeight = metrics.descent - metrics.ascent

        val rectLeft = out.width - textWidth - padding * 2
        val rectTop = out.height - textHeight - padding * 2
        val rectRight = out.width.toFloat()
        val rectBottom = out.height.toFloat()

        canvas.drawRect(rectLeft, rectTop, rectRight, rectBottom, bgPaint)
        canvas.drawText(
            label,
            rectLeft + padding,
            rectBottom - padding - metrics.descent,
            textPaint,
        )
        return out
    }
}
