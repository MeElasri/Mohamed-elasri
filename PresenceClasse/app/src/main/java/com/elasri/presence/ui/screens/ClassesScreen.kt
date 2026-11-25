package com.elasri.presence.ui.screens

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Logout
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.BarChart
import androidx.compose.material.icons.filled.DateRange
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Groups
import androidx.compose.material.icons.filled.Key
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.Person
import androidx.compose.material.icons.filled.Print
import androidx.compose.material.icons.filled.Share
import androidx.compose.material.icons.filled.UploadFile
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.elasri.presence.data.ClassImportResult
import com.elasri.presence.data.SchoolClass
import com.elasri.presence.data.UserProfile
import com.elasri.presence.ui.ClassesViewModel
import com.elasri.presence.ui.factoryOf
import com.elasri.presence.ui.repository
import com.elasri.presence.util.STUDENT_LIST_MIME_TYPES
import com.elasri.presence.util.readPickedFile
import com.elasri.presence.util.schoolYearLabel
import kotlinx.coroutines.launch
import java.time.LocalDate

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ClassesScreen(profile: UserProfile, onOpen: (String) -> Unit, onTeachers: () -> Unit, onStats: () -> Unit) {
    val repo = repository()
    val vm: ClassesViewModel = viewModel(factory = factoryOf { ClassesViewModel(repo) })
    val classes by vm.classes.collectAsStateWithLifecycle()
    val admin = profile.isAdmin

    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val snackbar = remember { SnackbarHostState() }
    val sender = rememberAdminSender()
    val printer = rememberAdminPrinter()

    var editor by remember { mutableStateOf<SchoolClass?>(null) }
    var toDelete by remember { mutableStateOf<SchoolClass?>(null) }
    var menuOpen by remember { mutableStateOf(false) }
    var pickReportDate by remember { mutableStateOf(false) }
    var pickPrintDate by remember { mutableStateOf(false) }
    var confirmLogout by remember { mutableStateOf(false) }
    var changingPassword by remember { mutableStateOf(false) }
    var importing by remember { mutableStateOf(false) }
    var importReport by remember { mutableStateOf<Pair<List<ClassImportResult>, List<String>>?>(null) }

    val massarLauncher = rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
        if (uris.isNotEmpty()) scope.launch {
            importing = true
            val files = mutableListOf<Pair<String, ByteArray>>()
            val readErrors = mutableListOf<String>()
            for (uri in uris) {
                try {
                    files += context.readPickedFile(uri)
                } catch (e: Exception) {
                    readErrors += "${uri.lastPathSegment ?: "fichier"} : ${e.message ?: "illisible"}"
                }
            }
            val (results, errors) = vm.importFiles(files)
            importing = false
            importReport = results to (readErrors + errors)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text(if (admin) "Classes" else "Mes classes")
                        Text(
                            profile.name.ifBlank { profile.email } + if (admin) " · Administration" else "",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                },
                actions = {
                    if (admin) IconButton(onClick = onStats) {
                        Icon(Icons.Filled.BarChart, contentDescription = "Statistiques de la semaine")
                    }
                    IconButton(onClick = { sender.send { repo.adminReportForDay(LocalDate.now()) } }) {
                        Icon(Icons.Filled.Share, contentDescription = "Envoyer les absences du jour")
                    }
                    IconButton(onClick = { menuOpen = true }, enabled = !importing) {
                        Icon(Icons.Filled.MoreVert, contentDescription = "Plus")
                    }
                    DropdownMenu(expanded = menuOpen, onDismissRequest = { menuOpen = false }) {
                        if (admin) {
                            DropdownMenuItem(
                                text = { Text("Statistiques de la semaine") },
                                leadingIcon = { Icon(Icons.Filled.BarChart, contentDescription = null) },
                                onClick = { menuOpen = false; onStats() }
                            )
                            DropdownMenuItem(
                                text = { Text("Importer depuis Massar (.xlsx)") },
                                leadingIcon = { Icon(Icons.Filled.UploadFile, contentDescription = null) },
                                onClick = { menuOpen = false; massarLauncher.launch(STUDENT_LIST_MIME_TYPES) }
                            )
                            DropdownMenuItem(
                                text = { Text("Enseignants et classes attribuées") },
                                leadingIcon = { Icon(Icons.Filled.Groups, contentDescription = null) },
                                onClick = { menuOpen = false; onTeachers() }
                            )
                            HorizontalDivider()
                        }
                        DropdownMenuItem(
                            text = { Text(if (admin) "Absences du jour (toutes les classes)" else "Envoyer mes absences du jour") },
                            leadingIcon = { Icon(Icons.Filled.Share, contentDescription = null) },
                            onClick = {
                                menuOpen = false
                                sender.send { repo.adminReportForDay(LocalDate.now()) }
                            }
                        )
                        DropdownMenuItem(
                            text = { Text("Absences d'une autre date") },
                            leadingIcon = { Icon(Icons.Filled.DateRange, contentDescription = null) },
                            onClick = { menuOpen = false; pickReportDate = true }
                        )
                        HorizontalDivider()
                        DropdownMenuItem(
                            text = { Text(if (admin) "Imprimer les absences du jour" else "Imprimer mes absences du jour") },
                            leadingIcon = { Icon(Icons.Filled.Print, contentDescription = null) },
                            onClick = { menuOpen = false; printer.print { repo.adminReportForDay(LocalDate.now()) } }
                        )
                        DropdownMenuItem(
                            text = { Text("Imprimer une autre date") },
                            leadingIcon = { Icon(Icons.Filled.Print, contentDescription = null) },
                            onClick = { menuOpen = false; pickPrintDate = true }
                        )
                        HorizontalDivider()
                        DropdownMenuItem(
                            text = { Text("Changer le mot de passe") },
                            leadingIcon = { Icon(Icons.Filled.Key, contentDescription = null) },
                            onClick = { menuOpen = false; changingPassword = true }
                        )
                        DropdownMenuItem(
                            text = { Text("Se déconnecter") },
                            leadingIcon = { Icon(Icons.AutoMirrored.Filled.Logout, contentDescription = null) },
                            onClick = { menuOpen = false; confirmLogout = true }
                        )
                    }
                }
            )
        },
        snackbarHost = { SnackbarHost(snackbar) },
        floatingActionButton = {
            if (admin) {
                ExtendedFloatingActionButton(
                    onClick = { editor = SchoolClass(id = "", name = "", year = schoolYearLabel()) },
                    icon = { Icon(Icons.Filled.Add, contentDescription = null) },
                    text = { Text("Nouvelle classe") }
                )
            }
        }
    ) { pad ->
        val list = classes
        when {
            importing || list == null -> Box(Modifier.padding(pad).fillMaxSize(), contentAlignment = Alignment.Center) {
                CircularProgressIndicator()
            }
            list.isEmpty() -> EmptyState(
                icon = Icons.Filled.Person,
                title = "Aucune classe",
                subtitle = if (admin)
                    "Menu ⋮ → « Importer depuis Massar » : chaque liste Excel devient une classe.\n" +
                        "Ensuite, menu ⋮ → « Enseignants » pour attribuer les classes."
                else
                    "Aucune classe ne vous est encore attribuée.\nContactez l'administration de l'établissement.",
                modifier = Modifier.padding(pad)
            )
            else -> LazyColumn(
                modifier = Modifier.padding(pad).fillMaxSize(),
                contentPadding = PaddingValues(start = 16.dp, end = 16.dp, top = 8.dp, bottom = 96.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                items(list, key = { it.id }) { c ->
                    ElevatedCard(onClick = { onOpen(c.id) }, modifier = Modifier.fillMaxWidth()) {
                        Row(
                            modifier = Modifier.padding(start = 16.dp, top = 14.dp, bottom = 14.dp, end = 4.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column(Modifier.weight(1f)) {
                                Text(c.name, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                                val meta = listOf(c.level, c.year).filter { it.isNotBlank() }.joinToString(" · ")
                                if (meta.isNotBlank()) {
                                    Text(meta, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                }
                            }
                            if (admin) {
                                IconButton(onClick = { editor = c }) { Icon(Icons.Filled.Edit, contentDescription = "Modifier") }
                                IconButton(onClick = { toDelete = c }) { Icon(Icons.Filled.Delete, contentDescription = "Supprimer") }
                            }
                        }
                    }
                }
            }
        }
    }

    if (pickReportDate) {
        DatePickDialog(initial = LocalDate.now(), onDismiss = { pickReportDate = false }) { date ->
            sender.send { repo.adminReportForDay(date) }
        }
    }

    if (pickPrintDate) {
        DatePickDialog(initial = LocalDate.now(), onDismiss = { pickPrintDate = false }) { date ->
            printer.print { repo.adminReportForDay(date) }
        }
    }

    editor?.let { c ->
        ClassEditorDialog(
            initial = c,
            onDismiss = { editor = null },
            onSave = { updated ->
                scope.launch {
                    val err = vm.save(updated)
                    if (err == null) editor = null else snackbar.showSnackbar(err)
                }
            }
        )
    }

    toDelete?.let { c ->
        ConfirmDialog(
            title = "Supprimer « ${c.name} » ?",
            text = "La classe et sa liste d'élèves seront supprimées et retirées des enseignants. " +
                "Les appels déjà enregistrés restent dans l'historique.",
            onConfirm = { scope.launch { vm.delete(c)?.let { snackbar.showSnackbar(it) } } },
            onDismiss = { toDelete = null }
        )
    }

    if (changingPassword) {
        ChangePasswordDialog(
            onDismiss = { changingPassword = false },
            onSubmit = { cur, new, conf ->
                vm.changePassword(cur, new, conf).also { err ->
                    if (err == null) scope.launch { snackbar.showSnackbar("Mot de passe modifié.") }
                }
            }
        )
    }

    if (confirmLogout) {
        ConfirmDialog(
            title = "Se déconnecter ?",
            text = "Les appels pas encore envoyés restent sur le téléphone et partiront à la prochaine connexion de ce compte.",
            confirmLabel = "Se déconnecter",
            onConfirm = { vm.signOut() },
            onDismiss = { confirmLogout = false }
        )
    }

    importReport?.let { (results, errors) ->
        AlertDialog(
            onDismissRequest = { importReport = null },
            title = { Text(if (results.isEmpty()) "Import impossible" else "Import terminé") },
            text = {
                Column(
                    modifier = Modifier.verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    results.forEach { r ->
                        val what = if (r.isNew) "nouvelle classe" else "classe existante"
                        val dup = if (r.skipped > 0) ", ${r.skipped} déjà présent(s)" else ""
                        Text("• ${r.className} ($what) : ${r.added} élève(s) ajouté(s)$dup")
                    }
                    errors.forEach { e -> Text("⚠ $e", color = MaterialTheme.colorScheme.error) }
                    if (results.isNotEmpty()) {
                        Text(
                            "Prochaine étape : menu ⋮ → « Enseignants » pour attribuer ces classes.",
                            style = MaterialTheme.typography.bodySmall
                        )
                    }
                }
            },
            confirmButton = { TextButton(onClick = { importReport = null }) { Text("OK") } }
        )
    }
}

@Composable
private fun ChangePasswordDialog(
    onDismiss: () -> Unit,
    onSubmit: suspend (current: String, new: String, confirm: String) -> String?
) {
    val scope = rememberCoroutineScope()
    var current by remember { mutableStateOf("") }
    var new by remember { mutableStateOf("") }
    var confirm by remember { mutableStateOf("") }
    var show by remember { mutableStateOf(false) }
    var busy by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf<String?>(null) }
    val visual = if (show) VisualTransformation.None else PasswordVisualTransformation()
    val kb = KeyboardOptions(keyboardType = KeyboardType.Password)

    AlertDialog(
        onDismissRequest = { if (!busy) onDismiss() },
        title = { Text("Changer le mot de passe") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                OutlinedTextField(
                    value = current, onValueChange = { current = it; error = null },
                    label = { Text("Mot de passe actuel") }, singleLine = true,
                    visualTransformation = visual, keyboardOptions = kb
                )
                OutlinedTextField(
                    value = new, onValueChange = { new = it; error = null },
                    label = { Text("Nouveau mot de passe") }, singleLine = true,
                    supportingText = { Text("6 caractères minimum") },
                    visualTransformation = visual, keyboardOptions = kb
                )
                OutlinedTextField(
                    value = confirm, onValueChange = { confirm = it; error = null },
                    label = { Text("Confirmer le nouveau mot de passe") }, singleLine = true,
                    visualTransformation = visual, keyboardOptions = kb
                )
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Checkbox(checked = show, onCheckedChange = { show = it })
                    Text("Afficher les mots de passe")
                }
                error?.let { Text(it, color = MaterialTheme.colorScheme.error) }
            }
        },
        confirmButton = {
            TextButton(
                enabled = !busy && current.isNotEmpty() && new.isNotEmpty() && confirm.isNotEmpty(),
                onClick = {
                    scope.launch {
                        busy = true
                        val err = onSubmit(current, new, confirm)
                        busy = false
                        if (err == null) onDismiss() else error = err
                    }
                }
            ) { Text(if (busy) "Patientez…" else "Valider") }
        },
        dismissButton = { TextButton(onClick = onDismiss, enabled = !busy) { Text("Annuler") } }
    )
}

@Composable
private fun ClassEditorDialog(initial: SchoolClass, onDismiss: () -> Unit, onSave: (SchoolClass) -> Unit) {
    var name by remember(initial) { mutableStateOf(initial.name) }
    var level by remember(initial) { mutableStateOf(initial.level) }
    var year by remember(initial) { mutableStateOf(initial.year) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(if (initial.id.isBlank()) "Nouvelle classe" else "Modifier la classe") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedTextField(
                    value = name, onValueChange = { name = it },
                    label = { Text("Nom (ex. TCS-3)") }, singleLine = true,
                    keyboardOptions = KeyboardOptions(capitalization = KeyboardCapitalization.Characters)
                )
                OutlinedTextField(
                    value = level, onValueChange = { level = it },
                    label = { Text("Niveau / filière (facultatif)") }, singleLine = true
                )
                OutlinedTextField(
                    value = year, onValueChange = { year = it },
                    label = { Text("Année scolaire") }, singleLine = true
                )
            }
        },
        confirmButton = {
            TextButton(
                enabled = name.isNotBlank(),
                onClick = { onSave(initial.copy(name = name.trim(), level = level.trim(), year = year.trim())) }
            ) { Text("Enregistrer") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Annuler") } }
    )
}
