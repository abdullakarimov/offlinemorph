package com.offlinemorph.android.feature.models

import java.io.File
import java.security.MessageDigest

class LocalModelManager(
    private val modelsDirectory: File,
) : ModelManager {
    override suspend fun getStatus(): ModelStatus {
        if (!modelsDirectory.exists()) {
            modelsDirectory.mkdirs()
        }

        val (installedModels, missingModels) = ModelCatalog.requiredModels.partition(::installedAndValid)
        val (optionalInstalled, optionalMissing) = ModelCatalog.optionalModels.partition(::installedAndValid)
        val (featurePackInstalled, featurePackMissing) =
            ModelCatalog.featurePackModels.partition(::installedAndValid)

        return ModelStatus(
            ready = missingModels.isEmpty(),
            installedModels = installedModels,
            missingModels = missingModels,
            modelDirectoryPath = modelsDirectory.absolutePath,
            optionalInstalledModels = optionalInstalled + featurePackInstalled,
            optionalMissingModels = optionalMissing + featurePackMissing,
        )
    }

    /**
     * Returns true when the model file exists on disk AND its SHA-256 digest matches the
     * expected value in [ModelSpec.sha256] (when one is declared). Files with no declared
     * checksum pass validation if they exist.
     */
    private fun installedAndValid(spec: ModelSpec): Boolean {
        val file = File(modelsDirectory, spec.fileName)
        if (!file.isFile) return false
        val expected = spec.sha256 ?: return true
        // Hashing a 100+ MB model takes a noticeable fraction of a second, so the digest is
        // cached until the file changes on disk.
        val key = Triple(file.absolutePath, file.length(), file.lastModified())
        val digest = digestCache[key] ?: sha256Hex(file).also { digestCache[key] = it }
        return digest == expected.lowercase()
    }

    private companion object {
        val digestCache = java.util.concurrent.ConcurrentHashMap<Triple<String, Long, Long>, String>()
    }

    private fun sha256Hex(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered().use { stream ->
            val buf = ByteArray(8192)
            var read: Int
            while (stream.read(buf).also { read = it } != -1) {
                digest.update(buf, 0, read)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
