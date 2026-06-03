package com.offlinemorph.android.feature.models

import android.app.Application
import java.io.File

object ModelPaths {
    fun appModelsDirectory(application: Application): File {
        return File(application.filesDir, "models")
    }
}
