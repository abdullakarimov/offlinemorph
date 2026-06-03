package com.offlinemorph.android.feature.consent

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

/**
 * Shown on first launch. User must explicitly accept before using the app.
 * The dialog is not dismissible by tapping outside — [onDecline] closes the app.
 */
@Composable
fun ConsentDialog(
    onAccept: () -> Unit,
    onDecline: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = { /* non-dismissible */ },
        title = {
            Text(
                text = "Before You Continue",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
            )
        },
        text = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Text(
                    text = "Offline Morph uses on-device AI to swap faces in photos and videos.",
                    style = MaterialTheme.typography.bodyMedium,
                )
                Text(
                    text = "By continuing you confirm that you understand and agree to all of the following:",
                    style = MaterialTheme.typography.bodyMedium,
                )
                ConsentPoint("You will only swap faces of real people who have given their explicit consent.")
                ConsentPoint("You will not use this app to create deceptive, misleading, or non-consensual content.")
                ConsentPoint("You will not use outputs to defame, harass, or harm any person.")
                ConsentPoint("You are solely responsible for any content you create with this app.")
                ConsentPoint("All outputs are AI-generated synthetic media and will be labelled as such when saved.")
                Text(
                    text = "Misuse of this app may violate local laws. If you do not agree, please exit now.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                )
            }
        },
        confirmButton = {
            Button(onClick = onAccept) {
                Text("I Understand & Agree")
            }
        },
        dismissButton = {
            TextButton(onClick = onDecline) {
                Text("Exit App")
            }
        },
    )
}

@Composable
private fun ConsentPoint(text: String) {
    Text(
        text = "• $text",
        style = MaterialTheme.typography.bodyMedium,
    )
}
