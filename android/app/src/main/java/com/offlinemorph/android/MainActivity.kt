package com.offlinemorph.android

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.offlinemorph.android.ui.theme.OfflineMorphTheme
import org.opencv.android.OpenCVLoader

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        // Initialise OpenCV native libraries before any Mat/Imgproc calls.
        // OpenCVLoader.initLocal() bundles the .so directly — no separate OpenCV Manager needed.
        OpenCVLoader.initLocal()
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        setContent {
            OfflineMorphTheme {
                OfflineMorphApp()
            }
        }
    }
}
