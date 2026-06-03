package com.offlinemorph.android.feature.models

class StubModelManager : ModelManager {
    override suspend fun getStatus(): ModelStatus {
        return ModelStatus(
            ready = false,
            installedModels = emptyList(),
            missingModels = ModelCatalog.requiredModels,
            modelDirectoryPath = "<unset>",
        )
    }
}

