package com.offlinemorph.android.feature.models

data class ModelStatus(
    val ready: Boolean,
    val installedModels: List<ModelSpec>,
    val missingModels: List<ModelSpec>,
    val modelDirectoryPath: String,
    val optionalInstalledModels: List<ModelSpec> = emptyList(),
    val optionalMissingModels: List<ModelSpec> = emptyList(),
)

interface ModelManager {
    suspend fun getStatus(): ModelStatus
}
