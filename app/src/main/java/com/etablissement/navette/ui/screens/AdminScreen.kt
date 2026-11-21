package com.etablissement.navette.ui.screens

import android.widget.Toast
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.text.KeyboardOptions
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
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
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
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.etablissement.navette.data.model.Establishment
import com.etablissement.navette.data.model.Teacher
import com.etablissement.navette.ui.NavetteViewModel
import com.etablissement.navette.ui.formatDate

/**
 * Écran d'administration des licences, visible uniquement par un compte administrateur.
 * Permet de créer un établissement, de suspendre ou prolonger sa licence, de changer
 * le nombre d'enseignants autorisés et de renouveler son code d'accès.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AdminScreen(viewModel: NavetteViewModel) {
    val establishments by viewModel.establishments.collectAsStateWithLifecycle()
    var showCreate by remember { mutableStateOf(false) }
    var editSeats by remember { mutableStateOf<Establishment?>(null) }
    var confirmCode by remember { mutableStateOf<Establishment?>(null) }
    var showTeachersOf by remember { mutableStateOf<Establishment?>(null) }

    Scaffold(
        topBar = { TopAppBar(title = { Text("Licences des établissements") }) },
        floatingActionButton = {
            FloatingActionButton(onClick = { showCreate = true }) {
                Icon(Icons.Filled.Add, contentDescription = "Ajouter un établissement")
            }
        }
    ) { padding ->
        if (establishments.isEmpty()) {
            Column(modifier = Modifier.fillMaxSize().padding(padding)) {
                EmptyState(
                    message = "Aucun établissement.\nAppuyez sur + pour créer la licence d'un client."
                )
            }
        } else {
            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding)
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                items(establishments, key = { it.id }) { establishment ->
                    LicenseCard(
                        establishment = establishment,
                        onToggleActive = { viewModel.adminSetActive(establishment.id, it) },
                        onExtend = { months -> viewModel.adminExtend(establishment, months) },
                        onEditSeats = { editSeats = establishment },
                        onNewCode = { confirmCode = establishment },
                        onRecount = { viewModel.adminRecount(establishment.id) },
                        onShowTeachers = { showTeachersOf = establishment }
                    )
                }
                item { Spacer(Modifier.height(72.dp)) }
            }
        }
    }

    if (showCreate) {
        CreateLicenseDialog(
            onDismiss = { showCreate = false },
            onCreate = { name, city, seats, months ->
                viewModel.adminCreateEstablishment(name, city, seats, months)
                showCreate = false
            }
        )
    }

    editSeats?.let { establishment ->
        var seats by remember(establishment.id) { mutableStateOf(establishment.maxTeachers.toString()) }
        AlertDialog(
            onDismissRequest = { editSeats = null },
            title = { Text("Nombre d'enseignants autorisés") },
            text = {
                OutlinedTextField(
                    value = seats,
                    onValueChange = { seats = it.filter(Char::isDigit).take(4) },
                    label = { Text("Maximum d'enseignants") },
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number)
                )
            },
            confirmButton = {
                TextButton(
                    enabled = (seats.toIntOrNull() ?: 0) > 0,
                    onClick = {
                        viewModel.adminSetMaxTeachers(establishment.id, seats.toInt())
                        editSeats = null
                    }
                ) { Text("Enregistrer") }
            },
            dismissButton = { TextButton(onClick = { editSeats = null }) { Text("Annuler") } }
        )
    }

    showTeachersOf?.let { establishment ->
        TeachersDialog(
            establishment = establishment,
            viewModel = viewModel,
            onDismiss = { showTeachersOf = null }
        )
    }

    confirmCode?.let { establishment ->
        AlertDialog(
            onDismissRequest = { confirmCode = null },
            title = { Text("Nouveau code d'accès ?") },
            text = {
                Text(
                    "L'ancien code de « ${establishment.name} » ne fonctionnera plus pour les " +
                        "nouveaux enseignants. Ceux qui ont déjà rejoint l'établissement gardent leur accès."
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
private fun LicenseCard(
    establishment: Establishment,
    onToggleActive: (Boolean) -> Unit,
    onExtend: (Int) -> Unit,
    onEditSeats: () -> Unit,
    onNewCode: () -> Unit,
    onRecount: () -> Unit,
    onShowTeachers: () -> Unit
) {
    val context = LocalContext.current
    val clipboard = LocalClipboardManager.current
    val now = System.currentTimeMillis()
    val expired = establishment.expiresAtMillis <= now
    val valid = establishment.hasValidLicense(now)

    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = establishment.label(),
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = when {
                            !establishment.active -> "● Suspendue"
                            expired -> "● Expirée"
                            else -> "● Active"
                        },
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.SemiBold,
                        color = if (valid) Color(0xFF2E7D32) else MaterialTheme.colorScheme.error
                    )
                }
                Switch(checked = establishment.active, onCheckedChange = onToggleActive)
            }

            Spacer(Modifier.height(8.dp))

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
                text = "Enseignants : ${establishment.memberCount} / ${establishment.maxTeachers}",
                style = MaterialTheme.typography.bodyMedium
            )
            Text(
                text = if (establishment.expiresAtMillis == 0L) "Aucune date de fin définie"
                else (if (expired) "Expirée le " else "Valide jusqu'au ") +
                    formatDate(establishment.expiresAtMillis),
                style = MaterialTheme.typography.bodyMedium,
                color = if (expired) MaterialTheme.colorScheme.error
                else MaterialTheme.colorScheme.onSurface
            )

            Spacer(Modifier.height(4.dp))

            Row {
                TextButton(onClick = { onExtend(1) }) { Text("+1 mois") }
                TextButton(onClick = { onExtend(12) }) { Text("+1 an") }
                TextButton(onClick = onEditSeats) { Text("Places") }
            }
            Row {
                TextButton(onClick = onNewCode) {
                    Text(if (establishment.code.isBlank()) "Créer un code" else "Nouveau code")
                }
                TextButton(onClick = onRecount) { Text("Recompter") }
                TextButton(onClick = onShowTeachers) { Text("Enseignants") }
            }
        }
    }
}

/** Liste des enseignants rattachés à un établissement. */
@Composable
private fun TeachersDialog(
    establishment: Establishment,
    viewModel: NavetteViewModel,
    onDismiss: () -> Unit
) {
    var teachers by remember(establishment.id) { mutableStateOf<List<Teacher>?>(null) }
    LaunchedEffect(establishment.id) {
        viewModel.adminLoadTeachers(establishment.id) { teachers = it }
    }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(establishment.label()) },
        text = {
            val list = teachers
            when {
                list == null -> Text("Chargement…")
                list.isEmpty() -> Text("Aucun enseignant n'a encore rejoint cet établissement.")
                else -> LazyColumn(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    item {
                        Text(
                            text = "${list.size} enseignant${if (list.size > 1) "s" else ""}",
                            style = MaterialTheme.typography.labelLarge,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
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
        confirmButton = { TextButton(onClick = onDismiss) { Text("Fermer") } }
    )
}

@Composable
private fun CreateLicenseDialog(
    onDismiss: () -> Unit,
    onCreate: (name: String, city: String, seats: Int, months: Int) -> Unit
) {
    var name by remember { mutableStateOf("") }
    var city by remember { mutableStateOf("") }
    var seats by remember { mutableStateOf("30") }
    var months by remember { mutableStateOf("12") }
    val valid = name.isNotBlank() &&
        (seats.toIntOrNull() ?: 0) > 0 && (months.toIntOrNull() ?: 0) > 0

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
                OutlinedTextField(
                    value = seats,
                    onValueChange = { seats = it.filter(Char::isDigit).take(4) },
                    label = { Text("Nombre maximum d'enseignants") },
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number)
                )
                OutlinedTextField(
                    value = months,
                    onValueChange = { months = it.filter(Char::isDigit).take(3) },
                    label = { Text("Durée de la licence (mois)") },
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number)
                )
                Text(
                    text = "Un code d'accès sera généré automatiquement.",
                    style = MaterialTheme.typography.bodySmall
                )
            }
        },
        confirmButton = {
            TextButton(
                enabled = valid,
                onClick = { onCreate(name, city, seats.toInt(), months.toInt()) }
            ) { Text("Créer") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Annuler") } }
    )
}
