package com.offlinemorph.android.feature.consent

import android.content.Context

private const val PREFS_NAME = "offlinemorph_consent"
private const val KEY_ACCEPTED = "consent_accepted_v1"

object ConsentManager {
    fun hasAccepted(context: Context): Boolean =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getBoolean(KEY_ACCEPTED, false)

    fun markAccepted(context: Context) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putBoolean(KEY_ACCEPTED, true)
            .apply()
    }
}
