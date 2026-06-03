package com.offlinemorph.android.feature.models

import android.content.ContentResolver
import android.net.Uri
import android.provider.OpenableColumns
import java.io.File

class AndroidModelInstaller(
    private val contentResolver: ContentResolver,
    private val modelsDirectory: File,
) : ModelInstaller {
    override suspend fun install(uris: List<Uri>): ModelInstallResult {
        if (!modelsDirectory.exists()) {
            modelsDirectory.mkdirs()
        }

        val installedModels = mutableListOf<ModelSpec>()
        val skippedFiles = mutableListOf<String>()
        val failedFiles = mutableListOf<String>()

        uris.forEach { uri ->
            val displayName = resolveDisplayName(uri) ?: uri.lastPathSegment ?: "unknown"
            val modelSpec = ModelCatalog.requiredModels.firstOrNull { it.fileName == displayName }

            if (modelSpec == null) {
                skippedFiles += displayName
                return@forEach
            }

            val destinationFile = File(modelsDirectory, modelSpec.fileName)
            val copied = runCatching {
                contentResolver.openInputStream(uri).use { inputStream ->
                    requireNotNull(inputStream) { "Unable to open input stream for $displayName" }
                    destinationFile.outputStream().use { outputStream ->
                        inputStream.copyTo(outputStream)
                    }
                }
            }.isSuccess

            if (copied) {
                installedModels += modelSpec
            } else {
                failedFiles += displayName
            }
        }

        return ModelInstallResult(
            installedModels = installedModels,
            skippedFiles = skippedFiles,
            failedFiles = failedFiles,
        )
    }

    private fun resolveDisplayName(uri: Uri): String? {
        contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { cursor ->
                val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (nameIndex >= 0 && cursor.moveToFirst()) {
                    return cursor.getString(nameIndex)
                }
            }
        return null
    }
}
