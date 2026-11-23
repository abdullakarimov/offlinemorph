package com.offlinemorph.android.feature.hairmakeup

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.offlinemorph.android.core.ml.hairmakeup.FacialHairStyle
import com.offlinemorph.android.core.ml.hairmakeup.HairStyle
import com.offlinemorph.android.ui.components.ZoomableImage
import kotlin.math.roundToInt

private data class ColorPreset(val label: String, val hex: String)

private val HAIR_PRESETS = listOf(
    ColorPreset("Black",    "1a1a1a"),
    ColorPreset("Dk Brown", "3b2314"),
    ColorPreset("Brown",    "6b3a2a"),
    ColorPreset("Auburn",   "7b2d00"),
    ColorPreset("Lt Brown", "a0522d"),
    ColorPreset("Blonde",   "c8a96e"),
    ColorPreset("Red",      "c0392b"),
    ColorPreset("Pink",     "e91e8c"),
    ColorPreset("Platinum", "e8e0d0"),
    ColorPreset("Blue",     "1a3a8b"),
    ColorPreset("Purple",   "6a1a9a"),
)

private val LIP_PRESETS = listOf(
    ColorPreset("Nude",    "d4956a"),
    ColorPreset("Blush",   "f48fb1"),
    ColorPreset("Rose",    "e91e63"),
    ColorPreset("Red",     "c62828"),
    ColorPreset("Deep",    "8b0000"),
    ColorPreset("Berry",   "8b1a8b"),
    ColorPreset("Coral",   "e8603c"),
    ColorPreset("Peach",   "ffab91"),
)

private val HAIR_SECONDARY_PRESETS = listOf(
    ColorPreset("Honey",    "d9a75b"),
    ColorPreset("Blonde",   "e3cc95"),
    ColorPreset("Platinum", "eee8dc"),
    ColorPreset("Copper",   "b5562a"),
    ColorPreset("Rose",     "e58aa8"),
    ColorPreset("Lilac",    "b49ad8"),
    ColorPreset("Teal",     "2a8c8c"),
)

private val EYESHADOW_PRESETS = listOf(
    ColorPreset("Taupe",   "8b7363"),
    ColorPreset("Bronze",  "9c6b3c"),
    ColorPreset("Gold",    "c9a24a"),
    ColorPreset("Plum",    "6e3a5e"),
    ColorPreset("Smoky",   "3d3a40"),
    ColorPreset("Rose",    "c9787f"),
    ColorPreset("Navy",    "2c3e6b"),
)

private val EYELINER_PRESETS = listOf(
    ColorPreset("Black",   "141414"),
    ColorPreset("Brown",   "3b2617"),
    ColorPreset("Navy",    "1c2a4a"),
    ColorPreset("Plum",    "4a1f3d"),
)

private val BLUSH_PRESETS = listOf(
    ColorPreset("Pink",    "f08da0"),
    ColorPreset("Peach",   "f4a582"),
    ColorPreset("Coral",   "ef7a65"),
    ColorPreset("Rose",    "d46a7e"),
    ColorPreset("Berry",   "b04a6a"),
)

private val FACIAL_HAIR_COLOR_PRESETS = listOf(
    ColorPreset("Black",   "1a1512"),
    ColorPreset("Dk Brown","3b2a1e"),
    ColorPreset("Brown",   "6b4a32"),
    ColorPreset("Ginger",  "9c5a2c"),
    ColorPreset("Blonde",  "b8995e"),
    ColorPreset("Grey",    "8a8580"),
)

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun HairMakeupScreen(viewModel: HairMakeupViewModel = viewModel()) {
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
        Text("Hair & Makeup", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)

        if (uiState.hairMakeupState is HairMakeupUiState.Unavailable) {
            Card(modifier = Modifier.fillMaxWidth()) {
                Text("Hair & Makeup is coming in a future update.", modifier = Modifier.padding(16.dp))
            }
            return@Column
        }

        if (uiState.hairMakeupState is HairMakeupUiState.ModelNotReady) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
            ) {
                Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text(
                        "Face Parsing Model Not Downloaded",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.SemiBold,
                    )
                    Text(
                        "Hair & makeup requires the face_parsing.onnx model. Open Setup → \"Download All (incl. Optional)\" to get it, then tap Apply again.",
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }
            }
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
                    "Face segmentation and colour blending run entirely on-device. No images are uploaded.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSecondaryContainer,
                )
            }
        }

        OutlinedButton(
            onClick = { photoPicker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text(if (uiState.sourceUri != null) "Change Portrait" else "Pick Portrait")
        }

        // ── Hair colour ───────────────────────────────────────────────────────
        Text("Hair Colour", style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.SemiBold)
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            ColorChip(label = "Off", hex = null, selected = uiState.hairColorHex == null) {
                viewModel.onHairColorSelected(null)
            }
            HAIR_PRESETS.forEach { preset ->
                ColorChip(label = preset.label, hex = preset.hex, selected = uiState.hairColorHex == preset.hex) {
                    viewModel.onHairColorSelected(preset.hex)
                }
            }
        }

        if (uiState.hairColorHex != null) {
            Text("Hair Style", style = MaterialTheme.typography.labelMedium)
            FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                HairStyle.entries.forEach { style ->
                    FilterChip(
                        selected = uiState.hairStyle == style,
                        onClick = { viewModel.onHairStyleSelected(style) },
                        label = { Text(style.displayName) },
                    )
                }
            }
            if (uiState.hairStyle != HairStyle.SOLID) {
                Text(
                    if (uiState.hairStyle == HairStyle.OMBRE) "Tip Colour" else "Highlight Colour",
                    style = MaterialTheme.typography.labelMedium,
                )
                PresetRow(HAIR_SECONDARY_PRESETS, uiState.hairSecondaryColorHex, offLabel = "Auto",
                    onSelect = viewModel::onHairSecondaryColorSelected)
            }
        }

        // ── Makeup ────────────────────────────────────────────────────────────
        Text("Lip Colour", style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.SemiBold)
        PresetRow(LIP_PRESETS, uiState.lipColorHex, onSelect = viewModel::onLipColorSelected)

        Text("Eyeshadow", style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.SemiBold)
        PresetRow(EYESHADOW_PRESETS, uiState.eyeshadowColorHex, onSelect = viewModel::onEyeshadowColorSelected)

        Text("Eyeliner", style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.SemiBold)
        PresetRow(EYELINER_PRESETS, uiState.eyelinerColorHex, onSelect = viewModel::onEyelinerColorSelected)

        Text("Blush", style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.SemiBold)
        PresetRow(BLUSH_PRESETS, uiState.blushColorHex, onSelect = viewModel::onBlushColorSelected)

        // ── Facial hair ───────────────────────────────────────────────────────
        Text("Facial Hair", style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.SemiBold)
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            FacialHairStyle.entries.forEach { style ->
                FilterChip(
                    selected = uiState.facialHair == style,
                    onClick = { viewModel.onFacialHairSelected(style) },
                    label = { Text(style.displayName) },
                )
            }
        }
        if (uiState.facialHair != FacialHairStyle.NONE) {
            Text("Facial Hair Colour", style = MaterialTheme.typography.labelMedium)
            PresetRow(FACIAL_HAIR_COLOR_PRESETS, uiState.facialHairColorHex, offLabel = "Match",
                onSelect = viewModel::onFacialHairColorSelected)
        }

        // ── Intensity ─────────────────────────────────────────────────────────
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
                onClick = viewModel::runHairMakeup,
                enabled = uiState.sourceUri != null && !uiState.isWorking && uiState.hasAnyEffect,
                modifier = Modifier.weight(1f),
            ) { Text("Apply") }
            OutlinedButton(
                onClick = viewModel::clearResult,
                enabled = uiState.hairMakeupState !is HairMakeupUiState.Idle,
                modifier = Modifier.weight(1f),
            ) { Text("Clear") }
        }

        when (val state = uiState.hairMakeupState) {
            is HairMakeupUiState.Loading -> {
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
            is HairMakeupUiState.Error -> {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Text(
                        state.message,
                        modifier = Modifier.padding(16.dp),
                        color = MaterialTheme.colorScheme.error,
                    )
                }
            }
            is HairMakeupUiState.Success -> {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text("Result", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold)
                        Text("Pinch to zoom · drag to pan", style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Spacer(Modifier.height(8.dp))
                    }
                    ZoomableImage(
                        bitmap = state.resultBitmap,
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(min = 300.dp),
                    )
                    Spacer(Modifier.height(16.dp))
                }
            }
            else -> Unit
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun PresetRow(
    presets: List<ColorPreset>,
    selectedHex: String?,
    offLabel: String = "Off",
    onSelect: (String?) -> Unit,
) {
    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        ColorChip(label = offLabel, hex = null, selected = selectedHex == null) { onSelect(null) }
        presets.forEach { preset ->
            ColorChip(label = preset.label, hex = preset.hex, selected = selectedHex == preset.hex) {
                onSelect(preset.hex)
            }
        }
    }
}

@Composable
private fun ColorChip(label: String, hex: String?, selected: Boolean, onClick: () -> Unit) {
    val borderColor = if (selected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outline
    val borderWidth = if (selected) 2.dp else 1.dp

    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(2.dp),
        modifier = Modifier.clickable(onClick = onClick),
    ) {
        Box(
            modifier = Modifier
                .size(32.dp)
                .clip(CircleShape)
                .border(BorderStroke(borderWidth, borderColor), CircleShape)
                .background(
                    if (hex != null) Color(android.graphics.Color.parseColor("#$hex"))
                    else MaterialTheme.colorScheme.surfaceVariant
                ),
        ) {
            if (hex == null) {
                // "Off" swatch — diagonal line
                Box(
                    modifier = Modifier
                        .size(2.dp, 32.dp)
                        .align(Alignment.Center)
                        .background(MaterialTheme.colorScheme.onSurfaceVariant),
                )
            }
        }
        Text(label, style = MaterialTheme.typography.labelSmall)
    }
}

