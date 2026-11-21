package com.etablissement.navette.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.etablissement.navette.data.model.Establishment
import com.etablissement.navette.ui.NavetteViewModel
import com.etablissement.navette.ui.formatDate
import com.etablissement.navette.ui.licenseProblem

/**
 * État de l'abonnement de l'établissement, en lecture seule.
 * C'est ce que voient les enseignants ; la gestion des licences (AdminScreen)
 * reste réservée au compte administrateur.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SubscriptionScreen(viewModel: NavetteViewModel) {
    val myProfile by viewModel.myProfile.collectAsStateWithLifecycle()
    val establishment by viewModel.myEstablishment.collectAsStateWithLifecycle()

    Scaffold(
        topBar = { TopAppBar(title = { Text("Abonnement") }) }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            val current = establishment
            if (current == null) {
                EmptyState(
                    message = licenseProblem(myProfile, null)
                        ?: "Chargement de l'état de l'abonnement…"
                )
            } else {
                SubscriptionCard(current)
                Text(
                    text = "Pour renouveler l'abonnement ou augmenter le nombre d'enseignants, " +
                        "l'administration de votre établissement doit contacter le fournisseur " +
                        "de l'application.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

@Composable
private fun SubscriptionCard(establishment: Establishment) {
    val now = System.currentTimeMillis()
    val expired = establishment.expiresAtMillis <= now
    val valid = establishment.hasValidLicense(now)
    val daysLeft = ((establishment.expiresAtMillis - now + DAY_MILLIS - 1) / DAY_MILLIS).toInt()
    val endingSoon = valid && daysLeft <= 30
    val green = Color(0xFF2E7D32)

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = if (valid) MaterialTheme.colorScheme.primaryContainer
            else MaterialTheme.colorScheme.errorContainer
        )
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = establishment.label(),
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold
            )
            Text(
                text = when {
                    !establishment.active -> "● Abonnement suspendu"
                    expired -> "● Abonnement expiré"
                    else -> "● Abonnement actif"
                },
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
                color = if (valid) green else MaterialTheme.colorScheme.error
            )
        }
    }

    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            InfoRow(
                label = if (expired) "Expiré le" else "Valide jusqu'au",
                value = formatDate(establishment.expiresAtMillis),
                valueColor = if (expired) MaterialTheme.colorScheme.error else Color.Unspecified
            )
            if (!expired) {
                InfoRow(
                    label = "Jours restants",
                    value = "$daysLeft",
                    valueColor = if (endingSoon) MaterialTheme.colorScheme.error else Color.Unspecified
                )
            }
            InfoRow(
                label = "Enseignants inscrits",
                value = "${establishment.memberCount} / ${establishment.maxTeachers}"
            )
            if (establishment.maxTeachers > 0) {
                LinearProgressIndicator(
                    progress = {
                        (establishment.memberCount.toFloat() / establishment.maxTeachers).coerceIn(0f, 1f)
                    },
                    modifier = Modifier.fillMaxWidth()
                )
            }
        }
    }

    val warning = when {
        !establishment.active ->
            "L'abonnement est suspendu : la navette et les groupes ne sont plus accessibles."
        expired ->
            "L'abonnement a expiré : la navette et les groupes ne sont plus accessibles."
        endingSoon ->
            "L'abonnement se termine dans $daysLeft jour${if (daysLeft > 1) "s" else ""}. " +
                "Pensez à le faire renouveler."
        establishment.memberCount >= establishment.maxTeachers ->
            "Le nombre maximum d'enseignants est atteint : aucun nouveau collègue ne peut rejoindre."
        else -> null
    }
    if (warning != null) {
        Text(
            text = warning,
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.SemiBold,
            color = MaterialTheme.colorScheme.error
        )
    }
}

@Composable
private fun InfoRow(label: String, value: String, valueColor: Color = Color.Unspecified) {
    Row(modifier = Modifier.fillMaxWidth()) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.weight(1f)
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodyLarge,
            fontWeight = FontWeight.SemiBold,
            color = valueColor
        )
    }
}

private const val DAY_MILLIS = 86_400_000L
