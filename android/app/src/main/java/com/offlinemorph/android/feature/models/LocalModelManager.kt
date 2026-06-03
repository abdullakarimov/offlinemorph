package com.offlinemorph.android.feature.models

import java.io.File

class LocalModelManager(
    private val modelsDirectory: File,
) : ModelManager {
    override suspend fun getStatus(): ModelStatus {
        if (!modelsDirectory.exists()) {
            modelsDirectory.mkdirs()
        }

        val installedModels = ModelCatalog.requiredModels.filter { spec ->
            File(modelsDirectory, spec.fileName).isFile
        }
        val missingModels = ModelCatalog.requiredModels.filterNot { spec ->
            File(modelsDirectory, spec.fileName).isFile
        }
        val optionalInstalled = ModelCatalog.optionalModels.filter { spec ->
            File(modelsDirectory, spec.fileName).isFile
        }
        val optionalMissing = ModelCatalog.optionalModels.filterNot { spec ->
            File(modelsDirectory, spec.fileName).isFile
        }

        return ModelStatus(
            ready = missingModels.isEmpty(),
            installedModels = installedModels,
            missingModels = missingModels,
            modelDirectoryPath = modelsDirectory.absolutePath,
            optionalInstalledModels = optionalInstalled,
            optionalMissingModels = optionalMissing,
        )
    }
}
