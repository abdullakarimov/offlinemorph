package com.offlinemorph.android.feature.swap

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.CloudDownload
import androidx.compose.material.icons.filled.ErrorOutline
import androidx.compose.material.icons.filled.RadioButtonUnchecked
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun AiSetupScreen(viewModel: SwapViewModel = viewModel()) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    val aiPackPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenMultipleDocuments(),
        onResult = viewModel::installModels,
    )

    Scaffold { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Text(
                text = "Setup",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                text = "Download AI model files to enable on-device face swap and enhancement.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            // ── Device capability ─────────────────────────────────────────────
            SetupStatusCard(
                title = uiState.deviceCapabilityTitle,
                message = uiState.deviceCapabilityMessage,
            )

            // ── Model download list ───────────────────────────────────────────
            if (uiState.downloadItems.isNotEmpty()) {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(14.dp),
                        verticalArrangement = Arrangement.spacedBy(2.dp),
                    ) {
                        Text(
                            "AI Model Files",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold,
                        )
                        Spacer(Modifier.height(6.dp))

                        uiState.downloadItems.forEachIndexed { index, item ->
                            if (index > 0) HorizontalDivider(modifier = Modifier.padding(vertical = 4.dp))
                            ModelRow(item = item)
                        }
                    }
                }
            }

            // ── Status message ────────────────────────────────────────────────
            if (uiState.modelInstallMessage.isNotBlank()) {
                SetupStatusCard(
                    title = "Status",
                    message = uiState.modelInstallMessage,
                )
            }

            // ── Actions ───────────────────────────────────────────────────────
            Button(
                onClick = viewModel::downloadModels,
                enabled = !uiState.isWorking,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(if (uiState.isWorking) "Downloading…" else "Download Required Models")
            }

            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                OutlinedButton(
                    onClick = viewModel::downloadAllModels,
                    enabled = !uiState.isWorking,
                ) {
                    Text("Download All (incl. Optional)")
                }
                OutlinedButton(
                    onClick = { aiPackPicker.launch(arrayOf("*/*")) },
                    enabled = !uiState.isWorking,
                ) {
                    Text("Import AI Files")
                }
                OutlinedButton(onClick = viewModel::refreshModelStatus) {
                    Text("Refresh")
                }
                OutlinedButton(onClick = viewModel::refreshDeviceCapability) {
                    Text("Assess Device")
                }
            }

            // ── Directory info ────────────────────────────────────────────────
            if (uiState.modelDirectoryPath.isNotBlank()) {
                Text(
                    "Models directory: ${uiState.modelDirectoryPath}",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
private fun ModelRow(item: ModelDownloadItem) {
    val isRequired = item.spec.required
    Column(modifier = Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            // Status icon
            when (item.state) {
                is ModelItemState.Installed, is ModelItemState.Done ->
                    Icon(
                        Icons.Default.CheckCircle,
                        contentDescription = "Installed",
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(20.dp),
                    )
                is ModelItemState.Downloading ->
                    Icon(
                        Icons.Default.CloudDownload,
                        contentDescription = "Downloading",
                        tint = MaterialTheme.colorScheme.secondary,
                        modifier = Modifier.size(20.dp),
                    )
                is ModelItemState.Failed ->
                    Icon(
                        Icons.Default.ErrorOutline,
                        contentDescription = "Failed",
                        tint = MaterialTheme.colorScheme.error,
                        modifier = Modifier.size(20.dp),
                    )
                else ->
                    Icon(
                        Icons.Default.RadioButtonUnchecked,
                        contentDescription = "Not installed",
                        tint = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.size(20.dp),
                    )
            }

            Column(modifier = Modifier.weight(1f)) {
                Row(
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        item.spec.fileName,
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.Medium,
                    )
                    if (!isRequired) {
                        Text(
                            "optional",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
                Text(
                    item.spec.role,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            // State label
            val stateLabel = when (item.state) {
                is ModelItemState.Installed -> "Installed"
                is ModelItemState.Done      -> "Done"
                is ModelItemState.Failed    -> "Failed"
                is ModelItemState.Downloading -> {
                    val pct = item.state.fraction
                    if (pct < 0f) "…" else "${(pct * 100).toInt()}%"
                }
                else -> if (isRequired) "Required" else "—"
            }
            Text(
                stateLabel,
                style = MaterialTheme.typography.labelMedium,
                color = when (item.state) {
                    is ModelItemState.Installed, is ModelItemState.Done ->
                        MaterialTheme.colorScheme.primary
                    is ModelItemState.Failed ->
                        MaterialTheme.colorScheme.error
                    else -> MaterialTheme.colorScheme.onSurfaceVariant
                },
            )
        }

        // Progress bar shown while downloading
        if (item.state is ModelItemState.Downloading) {
            val fraction = item.state.fraction
            if (fraction < 0f) {
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
            } else {
                LinearProgressIndicator(progress = { fraction }, modifier = Modifier.fillMaxWidth())
            }
        }
    }
}

@Composable
private fun SetupStatusCard(
    title: String,
    message: String,
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Text(text = title, style = MaterialTheme.typography.titleMedium)
            Text(text = message, style = MaterialTheme.typography.bodyMedium)
        }
    }
}
