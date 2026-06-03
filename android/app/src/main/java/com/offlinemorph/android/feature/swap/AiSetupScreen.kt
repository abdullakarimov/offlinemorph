package com.offlinemorph.android.feature.swap

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
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
            )
            Text(
                text = "Manage device capability and AI engine files used by photo and video swap.",
                style = MaterialTheme.typography.bodyMedium,
            )

            SetupStatusCard(
                title = uiState.deviceCapabilityTitle,
                message = uiState.deviceCapabilityMessage,
            )

            SetupStatusCard(
                title = "AI Pack Status",
                message = buildString {
                    append(uiState.modelStatusMessage)
                    if (uiState.modelDirectoryPath.isNotBlank()) {
                        append("\nAI files directory: ")
                        append(uiState.modelDirectoryPath)
                    }
                },
            )

            SetupStatusCard(
                title = "AI Pack Setup",
                message = uiState.modelInstallMessage,
            )

            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Button(
                    onClick = { aiPackPicker.launch(arrayOf("*/*")) },
                    enabled = !uiState.isWorking,
                ) {
                    Text("Import AI Files")
                }
                Button(
                    onClick = viewModel::downloadModels,
                    enabled = !uiState.isWorking,
                ) {
                    Text("Download AI Files")
                }
                Button(onClick = viewModel::refreshModelStatus) {
                    Text("Refresh AI Pack")
                }
                Button(onClick = viewModel::refreshDeviceCapability) {
                    Text("Assess Device")
                }
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