package com.elasri.presence.ui.screens

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.List
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.DateRange
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.Person
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
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
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.elasri.presence.data.Student
import com.elasri.presence.ui.ClassDetailViewModel
import com.elasri.presence.ui.factoryOf
import com.elasri.presence.ui.repository
import com.elasri.presence.util.STUDENT_LIST_MIME_TYPES
import com.elasri.presence.util.readPickedFile
import com.elasri.presence.data.frMessage
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ClassDetailScreen(
    classId: String,
    onBack: () -> Unit,
    onTakeAttendance: () -> Unit,
    onHistory: () -> Unit,
    onReports: () -> Unit
) {
    val repo = repository()
    val vm: ClassDetailViewModel = viewModel(factory = factoryOf { ClassDetailViewModel(repo, classId) })
    val cls by vm.schoolClass.collectAsStateWithLifecycle()
    val students by vm.students.collectAsStateWithLifecycle()

    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val snackbar = remember { SnackbarHostState() }

    var editor by remember { mutableStateOf<Student?>(null) }
    var toDelete by remember { mutableStateOf<Student?>(null) }
    var menuOpen by remember { mutableStateOf(false) }
    var showCsvHelp by remember { mutableStateOf(false) }

    val importLauncher = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) scope.launch {
            val message = try {
                val (name, bytes) = context.readPickedFile(uri)
                val (added, skipped) = vm.importFile(bytes, name)
                when {
                    added == 0 && skipped == 0 -> "Aucun élève trouvé dans ce fichier."
                    skipped > 0 -> "$added élève(s) ajouté(s), $skipped doublon(s) ignoré(s)."
                    else -> "$added élève(s) ajouté(s)."
                }
            } catch (e: Exception) {
                "Import impossible : ${e.frMessage()}"
            }
            snackbar.showSnackbar(message)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(cls?.name ?: "") },
                navigationIcon = { BackButton(onBack) },
                actions = {
                    if (vm.isAdmin) IconButton(onClick = { menuOpen = true }) {
                        Icon(Icons.Filled.MoreVert, contentDescription = "Plus")
                    }
                    DropdownMenu(expanded = menuOpen, onDismissRequest = { menuOpen = false }) {
                        DropdownMenuItem(
                            text = { Text("Importer des élèves (Massar / Excel / CSV)") },
                            leadingIcon = { Icon(Icons.Filled.Add, contentDescription = null) },
                            onClick = {
                                menuOpen = false
                                importLauncher.launch(STUDENT_LIST_MIME_TYPES)
                            }
                        )
                        DropdownMenuItem(
                            text = { Text("Formats acceptés") },
                            leadingIcon = { Icon(Icons.Filled.Info, contentDescription = null) },
                            onClick = { menuOpen = false; showCsvHelp = true }
                        )
                    }
                }
            )
        },
        snackbarHost = { SnackbarHost(snackbar) },
        floatingActionButton = {
            if (vm.isAdmin) {
                FloatingActionButton(onClick = { editor = Student(id = "", name = "") }) {
                    Icon(Icons.Filled.Add, contentDescription = "Ajouter un élève")
                }
            }
        }
    ) { pad ->
        Column(Modifier.padding(pad).fillMaxSize()) {
            Column(
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Button(onClick = onTakeAttendance, modifier = Modifier.fillMaxWidth(), enabled = students.isNotEmpty()) {
                    Icon(Icons.Filled.Check, contentDescription = null)
                    Spacer(Modifier.width(8.dp))
                    Text("Faire l'appel")
                }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(onClick = onHistory, modifier = Modifier.weight(1f)) {
                        Icon(Icons.Filled.DateRange, contentDescription = null)
                        Spacer(Modifier.width(6.dp))
                        Text("Historique")
                    }
                    OutlinedButton(onClick = onReports, modifier = Modifier.weight(1f)) {
                        Icon(Icons.AutoMirrored.Filled.List, contentDescription = null)
                        Spacer(Modifier.width(6.dp))
                        Text("Rapports")
                    }
                }
                Text(
                    "${students.size} élève(s)",
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 8.dp)
                )
            }
            HorizontalDivider()

            if (students.isEmpty()) {
                EmptyState(
                    icon = Icons.Filled.Person,
                    title = "Aucun élève",
                    subtitle = if (vm.isAdmin)
                        "Ajoutez les élèves un par un avec +, ou importez la liste Massar depuis le menu ⋮."
                    else
                        "La liste des élèves n'a pas encore été ajoutée par l'administration."
                )
            } else {
                LazyColumn(contentPadding = PaddingValues(bottom = 96.dp)) {
                    itemsIndexed(students, key = { _, s -> s.id }) { index, s ->
                        ListItem(
                            leadingContent = {
                                Text(
                                    "${index + 1}",
                                    style = MaterialTheme.typography.labelLarge,
                                    color = MaterialTheme.colorScheme.primary
                                )
                            },
                            headlineContent = { Text(s.name) },
                            supportingContent = if (s.massarCode.isNotBlank()) {
                                { Text(s.massarCode) }
                            } else null,
                            trailingContent = if (!vm.isAdmin) null else {
                                { Row {
                                    IconButton(onClick = { editor = s }) {
                                        Icon(Icons.Filled.Edit, contentDescription = "Modifier")
                                    }
                                    IconButton(onClick = { toDelete = s }) {
                                        Icon(Icons.Filled.Delete, contentDescription = "Supprimer")
                                    }
                                } }
                            }
                        )
                        HorizontalDivider()
                    }
                }
            }
        }
    }

    editor?.let { s ->
        StudentEditorDialog(
            initial = s,
            onDismiss = { editor = null },
            onSave = { student, keepOpen ->
                scope.launch { vm.saveStudent(student)?.let { snackbar.showSnackbar(it) } }
                if (!keepOpen) editor = null
            }
        )
    }
    toDelete?.let { s ->
        ConfirmDialog(
            title = "Supprimer ${s.name} ?",
            text = "L'élève sera retiré de la liste de la classe. Les appels déjà enregistrés restent dans l'historique.",
            onConfirm = { scope.launch { vm.deleteStudent(s)?.let { snackbar.showSnackbar(it) } } },
            onDismiss = { toDelete = null }
        )
    }
    if (showCsvHelp) {
        AlertDialog(
            onDismissRequest = { showCsvHelp = false },
            title = { Text("Importer une liste d'élèves") },
            text = {
                Text(
                    "Fichiers acceptés : la liste Excel (.xlsx) téléchargée depuis Massar, " +
                        "un fichier Excel .xlsx ou un fichier .csv.\n\n" +
                        "Colonnes reconnues (français ou arabe) : Code Massar / رقم التلميذ, " +
                        "Nom / النسب, Prénom / الاسم, Nom et prénom.\n\n" +
                        "Les lignes de titre au-dessus du tableau sont ignorées. " +
                        "Les élèves déjà présents ne sont pas ajoutés deux fois.\n\n" +
                        "Ancien format .xls : ouvrez-le dans Excel et enregistrez-le en .xlsx."
                )
            },
            confirmButton = { TextButton(onClick = { showCsvHelp = false }) { Text("Compris") } }
        )
    }
}

@Composable
private fun StudentEditorDialog(
    initial: Student,
    onDismiss: () -> Unit,
    onSave: (Student, keepOpen: Boolean) -> Unit
) {
    val isNew = initial.id.isBlank()
    var name by remember(initial) { mutableStateOf(initial.name) }
    var code by remember(initial) { mutableStateOf(initial.massarCode) }
    var addedCount by remember(initial) { mutableStateOf(0) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(if (isNew) "Nouvel élève" else "Modifier l'élève") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedTextField(
                    value = name, onValueChange = { name = it },
                    label = { Text("Nom et prénom") }, singleLine = true,
                    keyboardOptions = KeyboardOptions(capitalization = KeyboardCapitalization.Words)
                )
                OutlinedTextField(
                    value = code, onValueChange = { code = it.uppercase() },
                    label = { Text("Code Massar (facultatif)") }, singleLine = true
                )
                if (addedCount > 0) {
                    Text(
                        "$addedCount élève(s) ajouté(s)",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.primary
                    )
                }
            }
        },
        confirmButton = {
            Row {
                if (isNew) {
                    TextButton(
                        enabled = name.isNotBlank(),
                        onClick = {
                            onSave(initial.copy(name = name.trim(), massarCode = code.trim()), true)
                            name = ""; code = ""; addedCount++
                        }
                    ) { Text("+ Suivant") }
                }
                TextButton(
                    enabled = name.isNotBlank(),
                    onClick = { onSave(initial.copy(name = name.trim(), massarCode = code.trim()), false) }
                ) { Text("Enregistrer") }
            }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Fermer") } }
    )
}
