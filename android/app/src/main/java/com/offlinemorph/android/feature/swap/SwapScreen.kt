package com.offlinemorph.android.feature.swap

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.Checkbox
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import com.offlinemorph.android.core.ml.FaceFilterMode
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel

@OptIn(ExperimentalLayoutApi::class, ExperimentalMaterial3Api::class)
@Composable
fun SwapScreen(viewModel: SwapViewModel = viewModel()) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    var showZoomDialog by remember { mutableStateOf(false) }

    // Derived values from the swap pipeline sealed state.
    val swapState = uiState.swapState
    val isSwapping = swapState is SwapUiState.Loading
    val outputBitmap = (swapState as? SwapUiState.Success)?.resultBitmap
    val swapStatusMessage = when (swapState) {
        is SwapUiState.Idle -> "Pick a source image and a target image to begin."
        is SwapUiState.Loading -> swapState.message
        is SwapUiState.Success -> "Swap completed successfully."
        is SwapUiState.Error -> "Swap failed: ${swapState.exception.message}"
    }

    val sourcePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia(),
        onResult = viewModel::onSourceSelected,
    )
    val targetPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia(),
        onResult = viewModel::onTargetSelected,
    )
    Scaffold { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Text(
                text = "Offline Morph",
                style = MaterialTheme.typography.headlineMedium,
            )
            Text(
                text = "On-device photo face swap. Use the Setup tab to manage AI files and device checks.",
                style = MaterialTheme.typography.bodyMedium,
            )

            ImageSlotCard(
                title = "Source Face",
                selectedUri = uiState.sourceUri,
                buttonLabel = "Pick Source Image",
                onPick = {
                    sourcePicker.launch(
                        PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                    )
                },
            )
            ImageSlotCard(
                title = "Target Image",
                selectedUri = uiState.targetUri,
                buttonLabel = "Pick Target Image",
                onPick = {
                    targetPicker.launch(
                        PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                    )
                },
            )

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
                            )
                        }
                    }
                    if (uiState.faceFilterMode == FaceFilterMode.MALE_ONLY || uiState.faceFilterMode == FaceFilterMode.FEMALE_ONLY) {
                        if (!uiState.isGenderModelAvailable) {
                            Text(
                                text = "Install genderage.onnx to enable gender filtering.",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.error,
                            )
                        }
                    }
                }
            }

            // Face selection strip — shown in SPECIFIC mode when ≥2 faces are detected.
            if (uiState.faceFilterMode == FaceFilterMode.SPECIFIC) {
                if (uiState.isDetectingFaces) {
                    Card(modifier = Modifier.fillMaxWidth()) {
                        Row(
                            modifier = Modifier.padding(16.dp),
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            CircularProgressIndicator(modifier = Modifier.size(20.dp))
                            Text(
                                text = "Detecting faces in target image\u2026",
                                style = MaterialTheme.typography.bodyMedium,
                            )
                        }
                    }
                } else if (uiState.detectedTargetFaces.size > 1) {
                    TargetFaceSelectionCard(
                        faces = uiState.detectedTargetFaces,
                        selectedIndex = uiState.selectedTargetFaceIndex,
                        onFaceSelected = viewModel::selectTargetFace,
                    )
                }
            }

            StatusCard(
                title = "Swap Status",
                message = swapStatusMessage,
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                val enhancerInteractive = !isSwapping && !uiState.isWorking
                Checkbox(
                    checked = uiState.isEnhancerEnabled,
                    onCheckedChange = viewModel::toggleEnhancer,
                    enabled = enhancerInteractive,
                )
                Text(
                    text = if (uiState.isEnhancerModelAvailable) {
                        "Enable AI Face Enhancer (Slower)"
                    } else {
                        "Enable AI Face Enhancer (install GFPGANv1.4.onnx in Setup first)"
                    },
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurface.copy(
                        alpha = if (enhancerInteractive) 1f else 0.38f,
                    ),
                )
            }

            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Button(
                    onClick = viewModel::startSwap,
                    enabled = uiState.canSwap,
                ) {
                    Text("Swap Face")
                }
                Button(
                    onClick = viewModel::saveOutputImage,
                    enabled = !uiState.isWorking && !isSwapping && outputBitmap != null,
                ) {
                    Text("Save Output")
                }
            }

            if (uiState.isWorking || isSwapping) {
                CircularProgressIndicator()
            }

            val outputBitmapInfo = uiState.outputBitmapInfo
            if (outputBitmapInfo != null) {
                StatusCard(
                    title = "Output",
                    message = outputBitmapInfo,
                )
            }

            if (outputBitmap != null) {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp),
                    ) {
                        Text(text = "Preview", style = MaterialTheme.typography.titleMedium)
                        Text(
                            text = "Tap image to zoom",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        Image(
                            bitmap = outputBitmap.asImageBitmap(),
                            contentDescription = "Current swap preview",
                            modifier = Modifier
                                .fillMaxWidth()
                                .clickable { showZoomDialog = true },
                        )
                    }
                }
            }

            if (showZoomDialog && outputBitmap != null) {
                ZoomPreviewDialog(
                    bitmap = outputBitmap,
                    onDismiss = { showZoomDialog = false },
                )
            }
        }

        val workingMessage = when {
            isSwapping -> swapStatusMessage
            else -> uiState.modelInstallMessage
        }
        if (uiState.isWorking || isSwapping) {
            AlertDialog(
                onDismissRequest = {},
                confirmButton = {},
                title = { Text("Working") },
                text = {
                    Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        CircularProgressIndicator(modifier = Modifier.size(28.dp))
                        Text(workingMessage)
                    }
                },
            )
        }
    }
}

@Composable
private fun ZoomPreviewDialog(
    bitmap: android.graphics.Bitmap,
    onDismiss: () -> Unit,
) {
    var zoom by remember { mutableFloatStateOf(1f) }

    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = {
            TextButton(onClick = onDismiss) {
                Text("Close")
            }
        },
        title = {
            Text("Zoom Preview")
        },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Text("Zoom: ${"%.1f".format(zoom)}x", fontWeight = FontWeight.Medium)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        TextButton(onClick = { zoom = (zoom - 0.25f).coerceAtLeast(1f) }) {
                            Text("-")
                        }
                        TextButton(onClick = { zoom = (zoom + 0.25f).coerceAtMost(4f) }) {
                            Text("+")
                        }
                    }
                }
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = "Zoomed output preview",
                    modifier = Modifier
                        .fillMaxWidth()
                        .size(320.dp)
                        .graphicsLayer {
                            scaleX = zoom
                            scaleY = zoom
                        },
                )
            }
        },
    )
}

@Composable
private fun ImageSlotCard(
    title: String,
    selectedUri: Uri?,
    buttonLabel: String,
    onPick: () -> Unit,
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Text(text = title, style = MaterialTheme.typography.titleMedium)
            Text(
                text = selectedUri?.toString() ?: "No image selected.",
                style = MaterialTheme.typography.bodyMedium,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
            )
            Button(onClick = onPick) {
                Text(buttonLabel)
            }
        }
    }
}

@Composable
private fun TargetFaceSelectionCard(
    faces: List<android.graphics.Bitmap>,
    selectedIndex: Int,
    onFaceSelected: (Int) -> Unit,
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = "Select face to replace (${faces.size} detected)",
                style = MaterialTheme.typography.titleMedium,
            )
            LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                itemsIndexed(faces) { index, bitmap ->
                    val isSelected = index == selectedIndex
                    Box(
                        modifier = Modifier
                            .size(72.dp)
                            .border(
                                border = BorderStroke(
                                    width = if (isSelected) 3.dp else 0.dp,
                                    color = MaterialTheme.colorScheme.primary,
                                ),
                            )
                            .clickable { onFaceSelected(index) },
                        contentAlignment = Alignment.Center,
                    ) {
                        Image(
                            bitmap = bitmap.asImageBitmap(),
                            contentDescription = "Face $index",
                            modifier = Modifier.fillMaxSize(),
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun StatusCard(
    title: String,
    message: String,
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(text = title, style = MaterialTheme.typography.titleMedium)
            Text(text = message, style = MaterialTheme.typography.bodyMedium)
        }
    }
}
