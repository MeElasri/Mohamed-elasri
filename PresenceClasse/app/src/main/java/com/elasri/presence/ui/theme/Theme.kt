package com.elasri.presence.ui.theme

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext

val PresentColor = Color(0xFF2E7D32)
val AbsentColor = Color(0xFFC62828)
val LateColor = Color(0xFFEF6C00)

private val LightScheme = lightColorScheme(
    primary = Color(0xFF1B5E8C),
    onPrimary = Color.White,
    primaryContainer = Color(0xFFCFE5FF),
    onPrimaryContainer = Color(0xFF001D33),
    secondary = Color(0xFF00897B),
    tertiary = Color(0xFFFFB300)
)

private val DarkScheme = darkColorScheme(
    primary = Color(0xFF9ACBFF),
    onPrimary = Color(0xFF003355),
    primaryContainer = Color(0xFF004A78),
    secondary = Color(0xFF4DB6AC),
    tertiary = Color(0xFFFFCA28)
)

@Composable
fun PresenceTheme(content: @Composable () -> Unit) {
    val dark = isSystemInDarkTheme()
    val context = LocalContext.current
    val scheme = when {
        Build.VERSION.SDK_INT >= Build.VERSION_CODES.S ->
            if (dark) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        dark -> DarkScheme
        else -> LightScheme
    }
    MaterialTheme(colorScheme = scheme, content = content)
}
