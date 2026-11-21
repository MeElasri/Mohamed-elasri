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
import com.etablissement.navette.data.model.Subscription
import com.etablissement.navette.ui.NavetteViewModel
import com.etablissement.navette.ui.formatDate

/**
 * État de l'abonnement personnel de l'enseignant, en lecture seule :
 * période d'essai, période payée, expiration ou suspension.
 * L'activation se fait par l'administrateur, après paiement.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SubscriptionScreen(viewModel: NavetteViewModel) {
    val subscription by viewModel.mySubscription.collectAsStateWithLifecycle()

    Scaffold(
        topBar = { TopAppBar(title = { Text("Mon abonnement") }) }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            val current = subscription
            if (current == null) {
                EmptyState(message = "Chargement de l'état de votre abonnement…")
            } else {
                SubscriptionDetails(current, viewModel.currentEmail.orEmpty())
            }
        }
    }
}

@Composable
private fun SubscriptionDetails(subscription: Subscription, email: String) {
    val now = System.currentTimeMillis()
    val hasAccess = subscription.hasAccess(now)
    val paid = subscription.paidUntilMillis > now
    val onTrial = subscription.active && !paid && subscription.onTrial(now)
    val until = subscription.validUntilMillis()
    val daysLeft = ((until - now + Subscription.DAY_MILLIS - 1) / Subscription.DAY_MILLIS).toInt()
    val endingSoon = hasAccess && daysLeft <= 7
    val green = Color(0xFF2E7D32)

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = if (hasAccess) MaterialTheme.colorScheme.primaryContainer
            else MaterialTheme.colorScheme.errorContainer
        )
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = when {
                    !subscription.active -> "● Abonnement suspendu"
                    paid -> "● Abonnement actif"
                    onTrial -> "● Essai gratuit en cours"
                    subscription.paidUntilMillis > 0L -> "● Abonnement expiré"
                    else -> "● Essai gratuit terminé"
                },
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
                color = when {
                    !hasAccess -> MaterialTheme.colorScheme.error
                    paid -> green
                    else -> MaterialTheme.colorScheme.primary
                }
            )
            Text(text = email, style = MaterialTheme.typography.bodyMedium)
        }
    }

    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            InfoRow(
                label = when {
                    paid -> "Payé jusqu'au"
                    onTrial -> "Essai jusqu'au"
                    subscription.paidUntilMillis > 0L -> "Expiré le"
                    else -> "Essai terminé le"
                },
                value = formatDate(until),
                valueColor = if (until <= now) MaterialTheme.colorScheme.error else Color.Unspecified
            )
            if (hasAccess) {
                InfoRow(
                    label = "Jours restants",
                    value = "$daysLeft",
                    valueColor = if (endingSoon) MaterialTheme.colorScheme.error else Color.Unspecified
                )
            }
        }
    }

    val warning = when {
        !subscription.active ->
            "Votre abonnement est suspendu : la navette et les groupes ne sont plus accessibles."
        !hasAccess ->
            "La navette et les groupes ne sont plus accessibles tant que l'abonnement n'est pas activé."
        endingSoon && onTrial ->
            "Votre essai gratuit se termine dans $daysLeft jour${if (daysLeft > 1) "s" else ""}. " +
                "Abonnez-vous pour continuer à utiliser l'application."
        endingSoon ->
            "Votre abonnement se termine dans $daysLeft jour${if (daysLeft > 1) "s" else ""}. " +
                "Pensez à le renouveler."
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

    Text(
        text = "Pour vous abonner ou renouveler, contactez le fournisseur de l'application et " +
            "indiquez-lui l'adresse e-mail ci-dessus. Votre accès est activé dès réception du paiement.",
        style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant
    )
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
