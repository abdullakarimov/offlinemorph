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

class AndroidModelDownloader(
    private val modelsDirectory: File,
) {
    suspend fun downloadMissingRequiredModels(
        onProgress: (String) -> Unit = {},
    ): ModelDownloadResult = withContext(Dispatchers.IO) {
        if (!modelsDirectory.exists()) {
            modelsDirectory.mkdirs()
        }

        val downloaded = mutableListOf<ModelSpec>()
        val skipped = mutableListOf<ModelSpec>()
        val failed = mutableListOf<String>()

        val allModels = ModelCatalog.allModels
        for ((index, spec) in allModels.withIndex()) {
            val destination = File(modelsDirectory, spec.fileName)
            if (destination.isFile) {
                skipped += spec
                onProgress("[${index + 1}/${allModels.size}] ${spec.fileName} already exists.")
                continue
            }

            if (spec.downloadUrls.isEmpty()) {
                failed += "${spec.fileName} (no download URLs configured)"
                onProgress("[${index + 1}/${allModels.size}] ${spec.fileName} failed: no URL.")
                continue
            }

            val label = if (spec.required) spec.fileName else "${spec.fileName} (optional)"
            onProgress("[${index + 1}/${allModels.size}] Downloading $label...")
            val ok = downloadWithMirrors(spec, destination, onProgress)
            if (ok) {
                downloaded += spec
                onProgress("[${index + 1}/${allModels.size}] Downloaded ${spec.fileName}.")
            } else {
                failed += spec.fileName
                onProgress("[${index + 1}/${allModels.size}] Failed ${spec.fileName}.")
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
        onProgress: (String) -> Unit,
    ): Boolean {
        for (url in spec.downloadUrls) {
            val ok = runCatching {
                downloadFile(url, destination)
                true
            }.getOrElse { false }
            if (ok) {
                return true
            }
            onProgress("Retrying ${spec.fileName} with next mirror...")
        }
        return false
    }

    private fun downloadFile(urlString: String, destination: File) {
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
            if (status !in 200..299) {
                error("HTTP $status")
            }

            connection.inputStream.use { input ->
                temp.outputStream().use { output ->
                    input.copyTo(output)
                }
            }

            if (destination.exists()) {
                destination.delete()
            }
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
