package com.offlinemorph.android.feature.aging

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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlin.math.roundToInt

@Composable
fun AgingScreen(viewModel: AgingViewModel = viewModel()) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    val sourcePicker = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia(),
    ) { uri -> viewModel.onSourceSelected(uri) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("AI Aging / De-Aging", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)

        when (val state = uiState.agingState) {
            is AgingUiState.Unavailable -> {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Text(
                        text = "Aging feature is coming in a future update.",
                        modifier = Modifier.padding(16.dp),
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }
                return@Column
            }
            else -> Unit
        }

        // Source image picker
        OutlinedButton(
            onClick = { sourcePicker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text(if (uiState.sourceUri != null) "Change Portrait" else "Pick Portrait")
        }

        // Age offset slider
        Column {
            val offsetLabel = when {
                uiState.ageOffsetYears > 0 -> "+${uiState.ageOffsetYears} years (aging)"
                uiState.ageOffsetYears < 0 -> "${uiState.ageOffsetYears} years (de-aging)"
                else -> "No age change"
            }
            Text("Age offset: $offsetLabel", style = MaterialTheme.typography.labelMedium)
            Slider(
                value = uiState.ageOffsetYears.toFloat(),
                onValueChange = { viewModel.onAgeOffsetChanged(it.roundToInt()) },
                valueRange = -50f..50f,
                steps = 99,
                modifier = Modifier.fillMaxWidth(),
            )
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("−50", style = MaterialTheme.typography.labelSmall)
                Text("0", style = MaterialTheme.typography.labelSmall)
                Text("+50", style = MaterialTheme.typography.labelSmall)
            }
        }

        // Intensity slider
        Column {
            Text(
                "Intensity: ${(uiState.intensity * 100).roundToInt()}%",
                style = MaterialTheme.typography.labelMedium,
            )
            Slider(
                value = uiState.intensity,
                onValueChange = viewModel::onIntensityChanged,
                valueRange = 0f..1f,
                modifier = Modifier.fillMaxWidth(),
            )
        }

        // Run / clear
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                onClick = viewModel::runAging,
                enabled = uiState.sourceUri != null && !uiState.isWorking,
                modifier = Modifier.weight(1f),
            ) { Text("Run Aging") }

            OutlinedButton(
                onClick = viewModel::clearResult,
                enabled = uiState.agingState !is AgingUiState.Idle,
                modifier = Modifier.weight(1f),
            ) { Text("Clear") }
        }

        // Pipeline state card
        when (val state = uiState.agingState) {
            is AgingUiState.Loading -> {
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
            is AgingUiState.Error -> {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Text(
                        text = state.message,
                        modifier = Modifier.padding(16.dp),
                        color = MaterialTheme.colorScheme.error,
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }
            }
            is AgingUiState.Success -> {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text("Result", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold)
                        Spacer(Modifier.height(8.dp))
                        Box(contentAlignment = Alignment.Center, modifier = Modifier.fillMaxWidth()) {
                            Image(
                                bitmap = state.resultBitmap.asImageBitmap(),
                                contentDescription = "Aged portrait result",
                                modifier = Modifier.fillMaxWidth(),
                            )
                        }
                    }
                }
            }
            else -> Unit
        }
    }
}
