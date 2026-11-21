package com.etablissement.navette.ui.screens

import android.widget.Toast
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
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.etablissement.navette.data.model.Establishment
import com.etablissement.navette.data.model.Subscription
import com.etablissement.navette.data.model.Teacher
import com.etablissement.navette.ui.NavetteViewModel
import com.etablissement.navette.ui.formatDate

private enum class SubscriptionFilter(val label: String) {
    ALL("Tous"), TRIAL("Essai"), PAID("Payés"), EXPIRED("Expirés")
}

/**
 * Écran d'administration, visible uniquement par un compte administrateur.
 *
 * - Abonnements : chaque enseignant a son propre abonnement ; après réception du
 *   paiement, l'administrateur lui ajoute des mois payés.
 * - Établissements : création des établissements et de leur code d'accès.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AdminScreen(viewModel: NavetteViewModel) {
    var tab by remember { mutableIntStateOf(0) }
    var showCreate by remember { mutableStateOf(false) }

    Scaffold(
        topBar = { TopAppBar(title = { Text("Administration") }) },
        floatingActionButton = {
            if (tab == 1) {
                FloatingActionButton(onClick = { showCreate = true }) {
                    Icon(Icons.Filled.Add, contentDescription = "Ajouter un établissement")
                }
            }
        }
    ) { padding ->
        Column(modifier = Modifier.fillMaxSize().padding(padding)) {
            TabRow(selectedTabIndex = tab) {
                Tab(selected = tab == 0, onClick = { tab = 0 }, text = { Text("Abonnements") })
                Tab(selected = tab == 1, onClick = { tab = 1 }, text = { Text("Établissements") })
            }
            if (tab == 0) SubscriptionsTab(viewModel) else EstablishmentsTab(viewModel)
        }
    }

    if (showCreate) {
        CreateEstablishmentDialog(
            onDismiss = { showCreate = false },
            onCreate = { name, city ->
                viewModel.adminCreateEstablishment(name, city)
                showCreate = false
            }
        )
    }
}

// ------------------------------------------------------------------ Abonnements

@Composable
private fun SubscriptionsTab(viewModel: NavetteViewModel) {
    val subscriptions by viewModel.allSubscriptions.collectAsStateWithLifecycle()
    val teachers by viewModel.allTeachers.collectAsStateWithLifecycle()
    var search by remember { mutableStateOf("") }
    var filter by remember { mutableStateOf(SubscriptionFilter.ALL) }
    var confirmEnd by remember { mutableStateOf<Subscription?>(null) }

    val now = System.currentTimeMillis()
    val teacherById = remember(teachers) { teachers.associateBy { it.id } }
    val rows = remember(subscriptions, teacherById, search, filter) {
        subscriptions
            .map { it to teacherById[it.id] }
            .filter { (subscription, teacher) ->
                val text = "${teacher?.name.orEmpty()} ${subscription.email} " +
                    teacher?.establishmentName.orEmpty()
                (search.isBlank() || text.contains(search.trim(), ignoreCase = true)) &&
                    when (filter) {
                        SubscriptionFilter.ALL -> true
                        SubscriptionFilter.TRIAL -> subscription.active && subscription.onTrial(now)
                        SubscriptionFilter.PAID -> subscription.active && subscription.paidUntilMillis > now
                        SubscriptionFilter.EXPIRED -> !subscription.hasAccess(now)
                    }
            }
            .sortedBy { (subscription, teacher) ->
                (teacher?.name?.ifBlank { null } ?: subscription.email).lowercase()
            }
    }

    Column(modifier = Modifier.fillMaxSize().padding(horizontal = 16.dp)) {
        Spacer(Modifier.height(8.dp))
        OutlinedTextField(
            value = search,
            onValueChange = { search = it },
            label = { Text("Rechercher (nom, e-mail, établissement)") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth()
        )
        Spacer(Modifier.height(8.dp))
        LazyRowChips(
            items = SubscriptionFilter.entries,
            isSelected = { it == filter },
            label = { it.label },
            onClick = { filter = it }
        )
        Text(
            text = "${rows.size} compte${if (rows.size > 1) "s" else ""}",
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(vertical = 6.dp)
        )

        if (rows.isEmpty()) {
            EmptyState(message = "Aucun compte ne correspond.")
        } else {
            LazyColumn(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                items(rows, key = { it.first.id }) { (subscription, teacher) ->
                    SubscriptionCard(
                        subscription = subscription,
                        teacher = teacher,
                        onAddMonths = { months -> viewModel.adminAddPaidMonths(subscription, months) },
                        onToggleActive = {
                            viewModel.adminSetSubscriptionActive(subscription.id, !subscription.active)
                        },
                        onEndPaid = { confirmEnd = subscription }
                    )
                }
                item { Spacer(Modifier.height(24.dp)) }
            }
        }
    }

    confirmEnd?.let { subscription ->
        AlertDialog(
            onDismissRequest = { confirmEnd = null },
            title = { Text("Annuler la période payée ?") },
            text = {
                Text(
                    "La période payée de ${subscription.email} prendra fin immédiatement. " +
                        "À utiliser en cas d'erreur de saisie ou de remboursement."
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    viewModel.adminEndPaidPeriod(subscription.id)
                    confirmEnd = null
                }) { Text("Annuler la période") }
            },
            dismissButton = { TextButton(onClick = { confirmEnd = null }) { Text("Retour") } }
        )
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun SubscriptionCard(
    subscription: Subscription,
    teacher: Teacher?,
    onAddMonths: (Int) -> Unit,
    onToggleActive: () -> Unit,
    onEndPaid: () -> Unit
) {
    val now = System.currentTimeMillis()
    val hasAccess = subscription.hasAccess(now)
    val paid = subscription.paidUntilMillis > now
    val status = when {
        !subscription.active -> "● Suspendu"
        paid -> "● Payé jusqu'au ${formatDate(subscription.paidUntilMillis)}"
        subscription.onTrial(now) -> "● Essai jusqu'au ${formatDate(subscription.trialEndMillis())}"
        subscription.paidUntilMillis > 0L -> "● Expiré le ${formatDate(subscription.paidUntilMillis)}"
        else -> "● Essai terminé le ${formatDate(subscription.trialEndMillis())}"
    }

    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = teacher?.name?.ifBlank { null } ?: "(profil non rempli)",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            Text(
                text = subscription.email,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Text(
                text = teacher?.establishmentName?.ifBlank { null } ?: "Aucun établissement",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(Modifier.height(4.dp))
            Text(
                text = status,
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.SemiBold,
                color = when {
                    !hasAccess -> MaterialTheme.colorScheme.error
                    paid -> Color(0xFF2E7D32)
                    else -> MaterialTheme.colorScheme.primary
                }
            )
            // FlowRow : les actions passent à la ligne sur les écrans étroits.
            FlowRow {
                TextButton(onClick = { onAddMonths(1) }) { Text("+1 mois") }
                TextButton(onClick = { onAddMonths(3) }) { Text("+3 mois") }
                TextButton(onClick = { onAddMonths(12) }) { Text("+1 an") }
                TextButton(onClick = onToggleActive) {
                    Text(if (subscription.active) "Suspendre" else "Réactiver")
                }
                if (paid) {
                    TextButton(onClick = onEndPaid) { Text("Annuler la période") }
                }
            }
        }
    }
}

// ---------------------------------------------------------------- Établissements

@Composable
private fun EstablishmentsTab(viewModel: NavetteViewModel) {
    val establishments by viewModel.establishments.collectAsStateWithLifecycle()
    val teachers by viewModel.allTeachers.collectAsStateWithLifecycle()
    var confirmCode by remember { mutableStateOf<Establishment?>(null) }
    var showTeachersOf by remember { mutableStateOf<Establishment?>(null) }

    val teachersByEstablishment = remember(teachers) {
        teachers.groupBy { it.establishmentId }
    }

    if (establishments.isEmpty()) {
        EmptyState(
            message = "Aucun établissement.\nAppuyez sur + pour en créer un et obtenir son code d'accès."
        )
    } else {
        LazyColumn(
            modifier = Modifier.fillMaxSize().padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            items(establishments, key = { it.id }) { establishment ->
                EstablishmentCard(
                    establishment = establishment,
                    teacherCount = teachersByEstablishment[establishment.id]?.size ?: 0,
                    onNewCode = { confirmCode = establishment },
                    onShowTeachers = { showTeachersOf = establishment }
                )
            }
            item { Spacer(Modifier.height(72.dp)) }
        }
    }

    showTeachersOf?.let { establishment ->
        val list = (teachersByEstablishment[establishment.id] ?: emptyList())
            .sortedBy { it.name.lowercase() }
        AlertDialog(
            onDismissRequest = { showTeachersOf = null },
            title = { Text(establishment.label()) },
            text = {
                if (list.isEmpty()) {
                    Text("Aucun enseignant n'a encore rejoint cet établissement.")
                } else {
                    LazyColumn(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        items(list, key = { it.id }) { teacher ->
                            Column {
                                Text(
                                    text = teacher.name.ifBlank { "(profil non rempli)" } +
                                        if (teacher.hasCar) "  •  voiture" else "",
                                    style = MaterialTheme.typography.bodyLarge,
                                    fontWeight = FontWeight.SemiBold
                                )
                                val details = listOf(teacher.subject, teacher.phone, teacher.email)
                                    .filter { it.isNotBlank() }
                                    .joinToString(" • ")
                                if (details.isNotBlank()) {
                                    Text(
                                        text = details,
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                }
                            }
                        }
                    }
                }
            },
            confirmButton = { TextButton(onClick = { showTeachersOf = null }) { Text("Fermer") } }
        )
    }

    confirmCode?.let { establishment ->
        AlertDialog(
            onDismissRequest = { confirmCode = null },
            title = { Text("Nouveau code d'accès ?") },
            text = {
                Text(
                    "L'ancien code de « ${establishment.name} » ne fonctionnera plus pour les " +
                        "nouveaux enseignants. Ceux qui ont déjà rejoint l'établissement y restent."
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    viewModel.adminRegenerateCode(establishment)
                    confirmCode = null
                }) { Text("Générer") }
            },
            dismissButton = { TextButton(onClick = { confirmCode = null }) { Text("Annuler") } }
        )
    }
}

@Composable
private fun EstablishmentCard(
    establishment: Establishment,
    teacherCount: Int,
    onNewCode: () -> Unit,
    onShowTeachers: () -> Unit
) {
    val context = LocalContext.current
    val clipboard = LocalClipboardManager.current

    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = establishment.label(),
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Code d'accès : ", style = MaterialTheme.typography.bodyMedium)
                Text(
                    text = establishment.code.ifBlank { "aucun" },
                    style = MaterialTheme.typography.bodyLarge,
                    fontFamily = FontFamily.Monospace,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.weight(1f)
                )
                if (establishment.code.isNotBlank()) {
                    IconButton(onClick = {
                        clipboard.setText(AnnotatedString(establishment.code))
                        Toast.makeText(context, "Code copié", Toast.LENGTH_SHORT).show()
                    }) {
                        Icon(Icons.Filled.ContentCopy, contentDescription = "Copier le code")
                    }
                }
            }
            Text(
                text = "$teacherCount enseignant${if (teacherCount > 1) "s" else ""} inscrit" +
                    if (teacherCount > 1) "s" else "",
                style = MaterialTheme.typography.bodyMedium
            )
            Row {
                TextButton(onClick = onShowTeachers) { Text("Enseignants") }
                TextButton(onClick = onNewCode) {
                    Text(if (establishment.code.isBlank()) "Créer un code" else "Nouveau code")
                }
            }
        }
    }
}

@Composable
private fun CreateEstablishmentDialog(
    onDismiss: () -> Unit,
    onCreate: (name: String, city: String) -> Unit
) {
    var name by remember { mutableStateOf("") }
    var city by remember { mutableStateOf("") }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Nouvel établissement") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedTextField(
                    value = name,
                    onValueChange = { name = it },
                    label = { Text("Nom de l'établissement") },
                    singleLine = true
                )
                OutlinedTextField(
                    value = city,
                    onValueChange = { city = it },
                    label = { Text("Ville") },
                    singleLine = true
                )
                Text(
                    text = "Un code d'accès sera généré automatiquement.",
                    style = MaterialTheme.typography.bodySmall
                )
            }
        },
        confirmButton = {
            TextButton(enabled = name.isNotBlank(), onClick = { onCreate(name, city) }) {
                Text("Créer")
            }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Annuler") } }
    )
}
