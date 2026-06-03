package com.offlinemorph.android.feature.models

import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

data class ModelDownloadResult(
    val downloadedModels: List<ModelSpec>,
    val skippedModels: List<ModelSpec>,
    val failedModels: List<String>,
)

/**
 * Callback emitted during a single file download.
 *
 * @param fileName      file being downloaded.
 * @param fraction      0f..1f progress; -1f when content-length is unknown.
 * @param fileIndex     0-based index of the current file.
 * @param totalFiles    total number of files in this batch.
 */
data class FileDownloadProgress(
    val fileName: String,
    val fraction: Float,
    val fileIndex: Int,
    val totalFiles: Int,
)

class AndroidModelDownloader(
    private val modelsDirectory: File,
) {
    /**
     * Downloads only the models in [ModelCatalog.requiredModels] that are not yet on disk.
     * Emits [onProgress] text messages and [onFileProgress] byte-level progress per file.
     */
    suspend fun downloadMissingRequiredModels(
        onProgress: (String) -> Unit = {},
        onFileProgress: (FileDownloadProgress) -> Unit = {},
    ): ModelDownloadResult = download(
        specs = ModelCatalog.requiredModels,
        onProgress = onProgress,
        onFileProgress = onFileProgress,
    )

    /**
     * Downloads all models in [ModelCatalog.allModels] (required + optional) that are not yet
     * on disk. Emits [onProgress] text messages and [onFileProgress] byte-level progress per file.
     */
    suspend fun downloadAllModels(
        onProgress: (String) -> Unit = {},
        onFileProgress: (FileDownloadProgress) -> Unit = {},
    ): ModelDownloadResult = download(
        specs = ModelCatalog.allModels,
        onProgress = onProgress,
        onFileProgress = onFileProgress,
    )

    private suspend fun download(
        specs: List<ModelSpec>,
        onProgress: (String) -> Unit,
        onFileProgress: (FileDownloadProgress) -> Unit,
    ): ModelDownloadResult = withContext(Dispatchers.IO) {
        if (!modelsDirectory.exists()) {
            modelsDirectory.mkdirs()
        }

        val downloaded = mutableListOf<ModelSpec>()
        val skipped = mutableListOf<ModelSpec>()
        val failed = mutableListOf<String>()

        for ((index, spec) in specs.withIndex()) {
            val destination = File(modelsDirectory, spec.fileName)
            if (destination.isFile) {
                skipped += spec
                onProgress("[${index + 1}/${specs.size}] ${spec.fileName} already exists.")
                onFileProgress(FileDownloadProgress(spec.fileName, 1f, index, specs.size))
                continue
            }

            if (spec.downloadUrls.isEmpty()) {
                failed += "${spec.fileName} (no download URLs configured)"
                onProgress("[${index + 1}/${specs.size}] ${spec.fileName} failed: no URL.")
                continue
            }

            val label = if (spec.required) spec.fileName else "${spec.fileName} (optional)"
            onProgress("[${index + 1}/${specs.size}] Downloading $label...")
            val ok = downloadWithMirrors(spec, destination, index, specs.size, onProgress, onFileProgress)
            if (ok) {
                downloaded += spec
                onProgress("[${index + 1}/${specs.size}] Downloaded ${spec.fileName}.")
                onFileProgress(FileDownloadProgress(spec.fileName, 1f, index, specs.size))
            } else {
                failed += spec.fileName
                onProgress("[${index + 1}/${specs.size}] Failed ${spec.fileName}.")
            }
        }

        ModelDownloadResult(
            downloadedModels = downloaded,
            skippedModels = skipped,
            failedModels = failed,
        )
    }

    private fun downloadWithMirrors(
        spec: ModelSpec,
        destination: File,
        fileIndex: Int,
        totalFiles: Int,
        onProgress: (String) -> Unit,
        onFileProgress: (FileDownloadProgress) -> Unit,
    ): Boolean {
        for (url in spec.downloadUrls) {
            val ok = runCatching {
                downloadFile(url, destination, spec.fileName, fileIndex, totalFiles, onFileProgress)
                true
            }.getOrElse { false }
            if (ok) return true
            onProgress("Retrying ${spec.fileName} with next mirror...")
        }
        return false
    }

    private fun downloadFile(
        urlString: String,
        destination: File,
        fileName: String,
        fileIndex: Int,
        totalFiles: Int,
        onFileProgress: (FileDownloadProgress) -> Unit,
    ) {
        val temp = File(destination.parentFile, "${destination.name}.part")
        var connection: HttpURLConnection? = null
        try {
            connection = (URL(urlString).openConnection() as HttpURLConnection).apply {
                connectTimeout = 20_000
                readTimeout = 60_000
                requestMethod = "GET"
                instanceFollowRedirects = true
            }
            connection.connect()

            val status = connection.responseCode
            if (status !in 200..299) error("HTTP $status")

            val contentLength = connection.contentLengthLong // -1 if unknown
            var bytesRead = 0L
            val buf = ByteArray(8_192)

            connection.inputStream.use { input ->
                temp.outputStream().use { output ->
                    var n: Int
                    while (input.read(buf).also { n = it } != -1) {
                        output.write(buf, 0, n)
                        bytesRead += n
                        val fraction = if (contentLength > 0) {
                            (bytesRead.toFloat() / contentLength.toFloat()).coerceIn(0f, 1f)
                        } else -1f
                        onFileProgress(FileDownloadProgress(fileName, fraction, fileIndex, totalFiles))
                    }
                }
            }

            if (destination.exists()) destination.delete()
            if (!temp.renameTo(destination)) {
                temp.copyTo(destination, overwrite = true)
                temp.delete()
            }
        } finally {
            connection?.disconnect()
            if (temp.exists() && (!destination.exists() || destination.length() == 0L)) {
                temp.delete()
            }
        }
    }
}
