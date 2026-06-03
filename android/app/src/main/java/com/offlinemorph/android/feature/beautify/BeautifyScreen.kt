package com.offlinemorph.android.feature.beautify

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
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
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import com.offlinemorph.android.ui.components.ZoomableImage
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlin.math.roundToInt

@Composable
fun BeautifyScreen(viewModel: BeautifyViewModel = viewModel()) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    val photoPicker = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia(),
    ) { uri -> viewModel.onSourceSelected(uri) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("Beautify", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)

        if (uiState.beautifyState is BeautifyUiState.Unavailable) {
            Card(modifier = Modifier.fillMaxWidth()) {
                Text(
                    "Beautify is coming in a future update.",
                    modifier = Modifier.padding(16.dp),
                    style = MaterialTheme.typography.bodyMedium,
                )
            }
            return@Column
        }

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer),
        ) {
            Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text(
                    "On-Device Processing",
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onSecondaryContainer,
                )
                Text(
                    "All enhancements run entirely on your device. No images are uploaded.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSecondaryContainer,
                )
            }
        }

        OutlinedButton(
            onClick = {
                photoPicker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
            },
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text(if (uiState.sourceUri != null) "Change Portrait" else "Pick Portrait")
        }

        // ── Skin Smoothing (implemented) ──────────────────────────────────────
        ControlSlider(
            label = "Skin Smoothing: ${(uiState.skinSmoothing * 100).roundToInt()}%",
            value = uiState.skinSmoothing,
            onValueChange = viewModel::onSkinSmoothingChanged,
        )

        // ── Controls pending model support (shown, not yet applied) ───────────
        ControlSlider(
            label = "Eye Enlarge: ${(uiState.eyeEnlarge * 100).roundToInt()}%  (coming soon)",
            value = uiState.eyeEnlarge,
            onValueChange = viewModel::onEyeEnlargeChanged,
            enabled = false,
        )
        ControlSlider(
            label = "Face Slim: ${(uiState.faceSlim * 100).roundToInt()}%  (coming soon)",
            value = uiState.faceSlim,
            onValueChange = viewModel::onFaceSlimChanged,
            enabled = false,
        )
        ControlSlider(
            label = "Teeth Whitening: ${(uiState.teethWhiten * 100).roundToInt()}%  (coming soon)",
            value = uiState.teethWhiten,
            onValueChange = viewModel::onTeethWhitenChanged,
            enabled = false,
        )

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                onClick = viewModel::runBeautify,
                enabled = uiState.sourceUri != null && !uiState.isWorking,
                modifier = Modifier.weight(1f),
            ) { Text("Apply") }
            OutlinedButton(
                onClick = viewModel::clearResult,
                enabled = uiState.beautifyState !is BeautifyUiState.Idle,
                modifier = Modifier.weight(1f),
            ) { Text("Clear") }
        }

        when (val state = uiState.beautifyState) {
            is BeautifyUiState.Loading -> {
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
            is BeautifyUiState.Error -> {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Text(
                        state.message,
                        modifier = Modifier.padding(16.dp),
                        color = MaterialTheme.colorScheme.error,
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }
            }
            is BeautifyUiState.Success -> {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text("Result", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold)
                        Text("Pinch to zoom · drag to pan", style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
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

@Composable
private fun ControlSlider(
    label: String,
    value: Float,
    onValueChange: (Float) -> Unit,
    enabled: Boolean = true,
) {
    Column {
        Text(
            label,
            style = MaterialTheme.typography.labelMedium,
            color = if (enabled) MaterialTheme.colorScheme.onSurface
            else MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = 0f..1f,
            modifier = Modifier.fillMaxWidth(),
            enabled = enabled,
        )
    }
}

