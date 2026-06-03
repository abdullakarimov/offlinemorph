package com.offlinemorph.android

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Surface
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.systemBars
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.offlinemorph.android.feature.aging.AgingScreen
import com.offlinemorph.android.feature.beautify.BeautifyScreen
import com.offlinemorph.android.feature.beautify.BeautifyViewModel
import com.offlinemorph.android.feature.consent.ConsentDialog
import com.offlinemorph.android.feature.consent.ConsentManager
import com.offlinemorph.android.feature.flags.FeatureFlags
import com.offlinemorph.android.feature.hairmakeup.HairMakeupScreen
import com.offlinemorph.android.feature.hairmakeup.HairMakeupViewModel
import com.offlinemorph.android.feature.swap.AiSetupScreen
import com.offlinemorph.android.feature.swap.SwapScreen
import com.offlinemorph.android.feature.swap.SwapViewModel
import com.offlinemorph.android.feature.videoswap.VideoSwapScreen

private val TABS = buildList {
    add("Photo Swap")
    add("Video Swap")
    if (FeatureFlags.agingEnabled) add("Aging")
    if (FeatureFlags.hairMakeupEnabled) add("Hair & Makeup")
    if (FeatureFlags.beautifyEnabled) add("Beautify")
    add("Setup")
}

@Composable
fun OfflineMorphApp() {
    val context = LocalContext.current
    var consentAccepted by remember {
        mutableStateOf(ConsentManager.hasAccepted(context))
    }

    if (!consentAccepted) {
        ConsentDialog(
            onAccept = {
                ConsentManager.markAccepted(context)
                consentAccepted = true
            },
            onDecline = {
                (context as? android.app.Activity)?.finish()
            },
        )
        return
    }

    Surface(modifier = Modifier.fillMaxSize()) {
        val swapViewModel: SwapViewModel = viewModel()
        val hairMakeupViewModel: HairMakeupViewModel = viewModel()
        val beautifyViewModel: BeautifyViewModel = viewModel()
        val swapUiState by swapViewModel.uiState.collectAsStateWithLifecycle()
        var selectedTab by remember { mutableIntStateOf(0) }
        Column(modifier = Modifier
            .fillMaxSize()
            .windowInsetsPadding(WindowInsets.systemBars)
        ) {
            TabRow(selectedTabIndex = selectedTab) {
                TABS.forEachIndexed { index, title ->
                    Tab(
                        selected = selectedTab == index,
                        onClick = { selectedTab = index },
                        text = { Text(title) },
                    )
                }
            }
            when (TABS.getOrNull(selectedTab)) {
                "Photo Swap"   -> SwapScreen(viewModel = swapViewModel)
                "Video Swap"   -> VideoSwapScreen()
                "Aging"        -> AgingScreen()
                "Hair & Makeup" -> HairMakeupScreen(viewModel = hairMakeupViewModel)
                "Beautify"     -> BeautifyScreen(viewModel = beautifyViewModel)
                "Setup"        -> AiSetupScreen(viewModel = swapViewModel)
            }
        }

        if (swapUiState.showMissingAiPackAlert) {
            AlertDialog(
                onDismissRequest = swapViewModel::dismissMissingAiPackAlert,
                confirmButton = {
                    TextButton(onClick = {
                        selectedTab = TABS.indexOf("Setup").coerceAtLeast(0)
                        swapViewModel.dismissMissingAiPackAlert()
                    }) {
                        Text("Open Setup")
                    }
                },
                dismissButton = {
                    TextButton(onClick = swapViewModel::dismissMissingAiPackAlert) {
                        Text("Later")
                    }
                },
                title = { Text("AI Files Needed") },
                text = {
                    Text("No AI engine files were found on this device. Open Setup to download or import them before starting a swap.")
                },
            )
        }
    }
}
