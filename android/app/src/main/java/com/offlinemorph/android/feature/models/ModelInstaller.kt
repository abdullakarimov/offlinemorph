package com.offlinemorph.android.feature.models

import android.net.Uri

data class ModelInstallResult(
    val installedModels: List<ModelSpec>,
    val skippedFiles: List<String>,
    val failedFiles: List<String>,
)

interface ModelInstaller {
    suspend fun install(uris: List<Uri>): ModelInstallResult
}
