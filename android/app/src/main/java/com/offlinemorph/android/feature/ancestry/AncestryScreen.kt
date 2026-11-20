package com.offlinemorph.android.feature.ancestry

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.offlinemorph.android.ui.components.ZoomableImage
import kotlin.math.roundToInt

@Composable
fun AncestryScreen(viewModel: AncestryViewModel = viewModel()) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    val imageOnly = PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)

    val faceAPicker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) {
        viewModel.onFaceASelected(it)
    }
    val faceBPicker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) {
        viewModel.onFaceBSelected(it)
    }
    val basePicker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) {
        viewModel.onBaseSelected(it)
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("Ancestry Blend", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)

        when (uiState.ancestryState) {
            is AncestryUiState.Unavailable -> {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Text(
                        text = "Ancestry blend is coming in a future update.",
                        modifier = Modifier.padding(16.dp),
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }
                return@Column
            }
            is AncestryUiState.ModelNotReady -> {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
                ) {
                    Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Text("Face Models Not Installed", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
                        Text(
                            "Ancestry blend uses the core face-swap models. Open the Setup tab and tap \"Download Required Models\".",
                            style = MaterialTheme.typography.bodyMedium,
                        )
                    }
                }
            }
            else -> Unit
        }

        // Synthetic-generation disclosure
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer),
        ) {
            Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text(
                    "On-Device AI Generation",
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onSecondaryContainer,
                )
                Text(
                    "Blends the facial identity of two people into one portrait, entirely on your device. " +
                        "No images are uploaded. The result is an AI-generated projection, not a genetic prediction.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSecondaryContainer,
                )
            }
        }

        HorizontalDivider()

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = { faceAPicker.launch(imageOnly) }, modifier = Modifier.weight(1f)) {
                Text(if (uiState.faceAUri != null) "Face A ✓" else "Pick Face A")
            }
            OutlinedButton(onClick = { faceBPicker.launch(imageOnly) }, modifier = Modifier.weight(1f)) {
                Text(if (uiState.faceBUri != null) "Face B ✓" else "Pick Face B")
            }
        }
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = { basePicker.launch(imageOnly) }, modifier = Modifier.weight(1f)) {
                Text(if (uiState.baseUri != null) "Base Portrait ✓" else "Pick Base Portrait (optional)")
            }
            if (uiState.baseUri != null) {
                TextButton(onClick = viewModel::clearBase) { Text("Reset") }
            }
        }
        Text(
            if (uiState.baseUri != null) "The blended face is rendered onto the base portrait."
            else "Without a base portrait, the blended face is rendered onto Face A's photo.",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        // Blend ratio slider
        Column {
            val aPercent = (uiState.blendRatio * 100).roundToInt()
            Text("Blend: $aPercent% Face A · ${100 - aPercent}% Face B", style = MaterialTheme.typography.labelMedium)
            Slider(
                value = uiState.blendRatio,
                onValueChange = viewModel::onBlendRatioChanged,
                valueRange = 0f..1f,
                modifier = Modifier.fillMaxWidth(),
            )
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("Face B", style = MaterialTheme.typography.labelSmall)
                Text("Face A", style = MaterialTheme.typography.labelSmall)
            }
        }

        // Intensity slider
        Column {
            Text("Intensity: ${(uiState.intensity * 100).roundToInt()}%", style = MaterialTheme.typography.labelMedium)
            Slider(
                value = uiState.intensity,
                onValueChange = viewModel::onIntensityChanged,
                valueRange = 0f..1f,
                modifier = Modifier.fillMaxWidth(),
            )
        }

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                onClick = viewModel::runAncestry,
                enabled = uiState.canRun,
                modifier = Modifier.weight(1f),
            ) { Text("Blend Faces") }

            OutlinedButton(
                onClick = viewModel::clearResult,
                enabled = uiState.ancestryState !is AncestryUiState.Idle && !uiState.isWorking,
                modifier = Modifier.weight(1f),
            ) { Text("Clear") }
        }

        when (val state = uiState.ancestryState) {
            is AncestryUiState.Loading -> {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Row(
                        modifier = Modifier.padding(16.dp),
                        horizontalArrangement = Arrangement.spacedBy(12.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        CircularProgressIndicator(modifier = Modifier.size(24.dp))
                        Text(state.message, style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
            is AncestryUiState.Error -> {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Text(
                        text = state.message,
                        modifier = Modifier.padding(16.dp),
                        color = MaterialTheme.colorScheme.error,
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }
            }
            is AncestryUiState.Success -> {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text("Result", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold)
                        Text(
                            "Pinch to zoom · drag to pan",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        Spacer(Modifier.height(8.dp))
                    }
                    ZoomableImage(bitmap = state.resultBitmap)
                    Spacer(Modifier.height(16.dp))
                }
            }
            else -> Unit
        }
    }
}
