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
import androidx.compose.runtime.mutableStateOf
import com.offlinemorph.android.ui.components.ZoomableImage
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import com.offlinemorph.android.core.ml.FaceFilterMode
import com.offlinemorph.android.feature.device.QualityProfile
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

            // ── Quality profile ───────────────────────────────────────────────
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(
                    modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 12.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Text("Quality", style = MaterialTheme.typography.titleSmall)
                    val profiles = QualityProfile.entries
                    SingleChoiceSegmentedButtonRow(modifier = Modifier.fillMaxWidth()) {
                        profiles.forEachIndexed { index, profile ->
                            SegmentedButton(
                                selected = uiState.qualityProfile == profile,
                                onClick = { viewModel.setQualityProfile(profile) },
                                enabled = !isSwapping && !uiState.isWorking,
                                shape = SegmentedButtonDefaults.itemShape(index, profiles.size),
                                label = { Text(profile.displayName) },
                            )
                        }
                    }
                    Text(uiState.qualityProfile.description, style = MaterialTheme.typography.bodySmall)
                    val missing = buildList {
                        if (uiState.qualityProfile != QualityProfile.FAST && !uiState.isParsingModelAvailable) {
                            add("face_parsing.onnx (occlusion mask)")
                        }
                        if (uiState.qualityProfile == QualityProfile.STUDIO && !uiState.isEnhancerModelAvailable) {
                            add("GFPGANv1.4.onnx (face restoration)")
                        }
                    }
                    if (missing.isNotEmpty()) {
                        Text(
                            "Not installed, skipped: ${missing.joinToString()}. Get them in Setup → Download All.",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
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
                            text = "Pinch to zoom · drag to pan",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        ZoomableImage(bitmap = outputBitmap)
                    }
                }
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
