package com.offlinemorph.android.feature.videoswap

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.offlinemorph.android.core.ml.FaceFilterMode
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun VideoSwapScreen(
    viewModel: VideoSwapViewModel = viewModel(),
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    val sourcePicker = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri: Uri? -> uri?.let { viewModel.onSourceSelected(it) } }

    val videoPicker = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri: Uri? -> uri?.let { viewModel.onVideoSelected(it) } }

    Scaffold { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(horizontal = 16.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            // ── Source face image ──────────────────────────────────────────────
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("Source Face", style = MaterialTheme.typography.titleMedium)
                    Text(
                        text = if (uiState.sourceUri != null) "✓ ${uiState.sourceUri}" else "No image selected.",
                        style = MaterialTheme.typography.bodySmall,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                    Button(
                        onClick = {
                            sourcePicker.launch(
                                PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                            )
                        },
                        modifier = Modifier.fillMaxWidth(),
                    ) { Text("Pick Source Face Image") }
                }
            }

            // ── Target video ───────────────────────────────────────────────────
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("Target Video", style = MaterialTheme.typography.titleMedium)
                    Text(
                        text = if (uiState.targetVideoUri != null) "✓ ${uiState.targetVideoUri}" else "No video selected.",
                        style = MaterialTheme.typography.bodySmall,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                    Button(
                        onClick = {
                            videoPicker.launch(
                                PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly)
                            )
                        },
                        modifier = Modifier.fillMaxWidth(),
                    ) { Text("Pick Target Video") }
                }
            }

            // ── Enhancer toggle ────────────────────────────────────────────────
            Card(modifier = Modifier.fillMaxWidth()) {
                Row(
                    modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Column {
                        Text("GFPGAN Enhancer", style = MaterialTheme.typography.titleSmall)
                        Text("Improves quality. Slower per-frame.", style = MaterialTheme.typography.bodySmall)
                    }
                    Checkbox(
                        checked = uiState.enhancerEnabled,
                        onCheckedChange = viewModel::setEnhancerEnabled,
                    )
                }
            }

            // ── Face filter mode selector ──────────────────────────────────────
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(
                    modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 12.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Text("Face Filter", style = MaterialTheme.typography.titleSmall)
                    SingleChoiceSegmentedButtonRow(modifier = Modifier.fillMaxWidth()) {
                        FaceFilterMode.entries.forEachIndexed { index, mode ->
                            SegmentedButton(
                                selected = uiState.faceFilterMode == mode,
                                onClick = { viewModel.setFaceFilterMode(mode) },
                                shape = SegmentedButtonDefaults.itemShape(index, FaceFilterMode.entries.size),
                                label = { Text(mode.displayName) },
                                enabled = !uiState.isWorking,
                            )
                        }
                    }
                    if (uiState.faceFilterMode == FaceFilterMode.MALE_ONLY || uiState.faceFilterMode == FaceFilterMode.FEMALE_ONLY) {
                        if (!uiState.isGenderModelAvailable) {
                            Text(
                                text = "Install genderage.onnx from Setup to enable gender filtering.",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.error,
                            )
                        }
                    }
                }
            }

            // ── Swap / Cancel button ──────────────────────────────────────
            Button(
                onClick = viewModel::startVideoSwap,
                enabled = uiState.canSwap,
                modifier = Modifier.fillMaxWidth(),
            ) { Text("Swap Video") }

            // ── Progress ───────────────────────────────────────────────────────
            val loadingState = uiState.swapState as? VideoSwapUiState.Loading
            if (loadingState != null) {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(modifier = Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            CircularProgressIndicator(modifier = Modifier.size(18.dp))
                            Text(loadingState.statusMessage, style = MaterialTheme.typography.bodyMedium)
                        }
                        if (loadingState.totalFrames > 0) {
                            LinearProgressIndicator(
                                progress = { loadingState.currentFrame.toFloat() / loadingState.totalFrames },
                                modifier = Modifier.fillMaxWidth(),
                            )
                            Text(
                                text = "${loadingState.currentFrame} / ${loadingState.totalFrames} frames",
                                style = MaterialTheme.typography.labelSmall,
                            )
                        }
                    }
                }
                // Dedicated stop button — visible only during active processing.
                Button(
                    onClick = viewModel::cancelVideoSwap,
                    modifier = Modifier.fillMaxWidth(),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.error,
                        contentColor  = MaterialTheme.colorScheme.onError,
                    ),
                ) { Text("Stop Processing") }
            }

            // ── Success ────────────────────────────────────────────────────────
            val successState = uiState.swapState as? VideoSwapUiState.Success
            if (successState != null) {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(modifier = Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text("Done!", style = MaterialTheme.typography.titleMedium)
                        Text(successState.outputFile.absolutePath, style = MaterialTheme.typography.bodySmall, maxLines = 2, overflow = TextOverflow.Ellipsis)
                        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            OutlinedButton(onClick = viewModel::saveOutput, modifier = Modifier.weight(1f)) { Text("Save to Gallery") }
                            Button(onClick = viewModel::shareOutput, modifier = Modifier.weight(1f)) { Text("Share") }
                        }
                    }
                }
            }

            // ── Error ──────────────────────────────────────────────────────────
            val errorState = uiState.swapState as? VideoSwapUiState.Error
            if (errorState != null) {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(modifier = Modifier.fillMaxWidth().padding(16.dp)) {
                        Text("Error", style = MaterialTheme.typography.titleMedium, color = MaterialTheme.colorScheme.error)
                        Text(errorState.message, style = MaterialTheme.typography.bodyMedium)
                    }
                }
            }
        }
    }
}
