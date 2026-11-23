package com.offlinemorph.android

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.offlinemorph.android.core.ml.NpuWarmup
import com.offlinemorph.android.feature.models.ModelPaths
import com.offlinemorph.android.ui.theme.OfflineMorphTheme
import org.opencv.android.OpenCVLoader

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        // Initialise OpenCV native libraries before any Mat/Imgproc calls.
        // OpenCVLoader.initLocal() bundles the .so directly — no separate OpenCV Manager needed.
        OpenCVLoader.initLocal()
        // Compile NPU graphs for installed models in the background (no-op when already cached
        // or on non-Snapdragon devices).
        NpuWarmup.start(ModelPaths.appModelsDirectory(application))
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        setContent {
            OfflineMorphTheme {
                OfflineMorphApp()
            }
        }
    }
}
