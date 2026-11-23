package com.elasri.presence.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.elasri.presence.data.BlockReason
import com.elasri.presence.data.LicenseState
import com.elasri.presence.util.fmtShort
import kotlinx.coroutines.launch
import java.time.Instant
import java.time.ZoneId

private val Navy = Color(0xFF13315C)
private val Grey = Color(0xFF5F6B7A)
private val Teal = Color(0xFF16888A)

/**
 * Blocking screen shown when the establishment's subscription is expired, suspended or missing.
 * The licence is renewed from the Firebase console (see ABONNEMENT.md); the app only reads it.
 */
@Composable
fun SubscriptionScreen(
    state: LicenseState.Blocked,
    onRetry: suspend () -> Unit,
    onSignOut: () -> Unit
) {
    val scope = rememberCoroutineScope()
    var busy by remember { mutableStateOf(false) }

    val establishment = state.license?.establishment.orEmpty()
    val lastValidDay = state.license?.expiresAt
        ?.takeIf { it > 0L }
        ?.let { Instant.ofEpochMilli(it - 1).atZone(ZoneId.systemDefault()).toLocalDate() }

    val message = when (state.reason) {
        BlockReason.EXPIRED ->
            "L'abonnement de l'établissement a expiré" +
                (lastValidDay?.let { " le ${it.fmtShort()}" } ?: "") + "."
        BlockReason.INACTIVE -> "L'abonnement de l'établissement a été suspendu."
        BlockReason.MISSING -> "Aucun abonnement actif n'est configuré pour cet établissement."
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.White)
            .verticalScroll(rememberScrollState())
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Icon(Icons.Filled.Lock, contentDescription = null, tint = Navy, modifier = Modifier.size(56.dp))
        Spacer(Modifier.height(16.dp))
        Text("Abonnement requis", color = Navy, fontSize = 24.sp, fontWeight = FontWeight.Bold)

        if (establishment.isNotBlank()) {
            Spacer(Modifier.height(4.dp))
            Text(establishment, color = Grey, fontSize = 14.sp, fontWeight = FontWeight.Medium)
        }

        Spacer(Modifier.height(14.dp))
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .background(Color(0xFFF2F5F9), RoundedCornerShape(12.dp))
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(message, color = Navy, textAlign = TextAlign.Center, fontWeight = FontWeight.Medium)
            Spacer(Modifier.height(8.dp))
            Text(
                "Pour continuer à utiliser l'application, contactez le fournisseur afin de renouveler " +
                    "l'abonnement. L'accès est rétabli automatiquement dès le renouvellement.",
                color = Grey, textAlign = TextAlign.Center, fontSize = 13.sp
            )
        }

        Spacer(Modifier.height(24.dp))
        Button(
            onClick = {
                if (!busy) scope.launch {
                    busy = true
                    runCatching { onRetry() }
                    busy = false
                }
            },
            enabled = !busy,
            shape = RoundedCornerShape(16.dp),
            colors = ButtonDefaults.buttonColors(containerColor = Teal, contentColor = Color.White),
            modifier = Modifier.fillMaxWidth().height(52.dp)
        ) {
            if (busy) {
                CircularProgressIndicator(color = Color.White, strokeWidth = 2.dp, modifier = Modifier.size(20.dp))
            } else {
                Icon(Icons.Filled.Refresh, contentDescription = null)
                Spacer(Modifier.width(8.dp))
                Text("Réessayer", fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
            }
        }
        Spacer(Modifier.height(10.dp))
        OutlinedButton(onClick = onSignOut, enabled = !busy, modifier = Modifier.fillMaxWidth()) {
            Text("Se déconnecter")
        }
    }
}
