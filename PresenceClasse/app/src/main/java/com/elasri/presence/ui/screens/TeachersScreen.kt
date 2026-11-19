package com.elasri.presence.ui.screens

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Groups
import androidx.compose.material.icons.filled.PersonAdd
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Switch
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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.elasri.presence.data.Role
import com.elasri.presence.data.SchoolClass
import com.elasri.presence.data.UserProfile
import com.elasri.presence.ui.TeachersViewModel
import com.elasri.presence.ui.factoryOf
import com.elasri.presence.ui.repository
import kotlinx.coroutines.launch
import java.security.SecureRandom

/** Administration: teacher accounts and the classes assigned to each one. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TeachersScreen(onBack: () -> Unit) {
    val repo = repository()
    val vm: TeachersViewModel = viewModel(factory = factoryOf { TeachersViewModel(repo) })
    val users by vm.users.collectAsStateWithLifecycle()
    val classes by vm.classes.collectAsStateWithLifecycle()
    val snackbar = remember { SnackbarHostState() }
    val scope = rememberCoroutineScope()

    var creating by remember { mutableStateOf(false) }
    var editing by remember { mutableStateOf<UserProfile?>(null) }

    Scaffold(
        topBar = {
            TopAppBar(title = { Text("Enseignants") }, navigationIcon = { BackButton(onBack) })
        },
        snackbarHost = { SnackbarHost(snackbar) },
        floatingActionButton = {
            ExtendedFloatingActionButton(
                onClick = { creating = true },
                icon = { Icon(Icons.Filled.PersonAdd, contentDescription = null) },
                text = { Text("Nouveau compte") }
            )
        }
    ) { pad ->
        val list = users
        when {
            list == null -> Box(Modifier.padding(pad).fillMaxSize(), contentAlignment = Alignment.Center) {
                CircularProgressIndicator()
            }
            list.isEmpty() -> EmptyState(
                icon = Icons.Filled.Groups,
                title = "Aucun compte",
                subtitle = "Créez un compte pour chaque enseignant, puis cochez ses classes.",
                modifier = Modifier.padding(pad)
            )
            else -> LazyColumn(Modifier.padding(pad).fillMaxSize(), contentPadding = PaddingValues(bottom = 96.dp)) {
                items(list, key = { it.uid }) { u ->
                    val names = classes.filter { it.id in u.classIds }.joinToString(", ") { it.name }
                    ListItem(
                        modifier = Modifier.clickable { editing = u },
                        overlineContent = {
                            Text(
                                u.role.label + if (!u.active) " · désactivé" else "",
                                color = if (u.active) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error
                            )
                        },
                        headlineContent = { Text(u.name.ifBlank { u.email }, fontWeight = FontWeight.Medium) },
                        supportingContent = {
                            Text(
                                when {
                                    u.role == Role.ADMIN -> "${u.email} · toutes les classes"
                                    names.isBlank() -> "${u.email} · aucune classe attribuée"
                                    else -> "${u.email}\n$names"
                                }
                            )
                        }
                    )
                    HorizontalDivider()
                }
            }
        }
    }

    if (creating) {
        UserEditorDialog(
            initial = null,
            classes = classes,
            isSelf = false,
            onDismiss = { creating = false },
            onSave = { name, email, password, role, ids, _ ->
                scope.launch {
                    val err = vm.create(name, email, password, role, ids)
                    if (err == null) {
                        creating = false
                        snackbar.showSnackbar("Compte créé. Communiquez l'e-mail et le mot de passe à $name.")
                    } else snackbar.showSnackbar(err)
                }
            }
        )
    }
    editing?.let { u ->
        UserEditorDialog(
            initial = u,
            classes = classes,
            isSelf = u.uid == vm.myUid,
            onDismiss = { editing = null },
            onSave = { name, _, _, role, ids, active ->
                scope.launch {
                    val err = vm.update(u, name, role, ids, active)
                    if (err == null) editing = null else snackbar.showSnackbar(err)
                }
            }
        )
    }
}

private fun randomPassword(): String {
    val chars = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnpqrstuvwxyz23456789"
    val rnd = SecureRandom()
    return (1..10).map { chars[rnd.nextInt(chars.length)] }.joinToString("")
}

@Composable
private fun UserEditorDialog(
    initial: UserProfile?,
    classes: List<SchoolClass>,
    isSelf: Boolean,
    onDismiss: () -> Unit,
    onSave: (name: String, email: String, password: String, role: Role, classIds: List<String>, active: Boolean) -> Unit
) {
    val isNew = initial == null
    var name by remember { mutableStateOf(initial?.name.orEmpty()) }
    var email by remember { mutableStateOf(initial?.email.orEmpty()) }
    var password by remember { mutableStateOf(randomPassword()) }
    var role by remember { mutableStateOf(initial?.role ?: Role.TEACHER) }
    var active by remember { mutableStateOf(initial?.active ?: true) }
    var selected by remember { mutableStateOf(initial?.classIds?.toSet() ?: emptySet()) }

    val valid = name.isNotBlank() && (!isNew || (email.contains('@') && password.length >= 6))

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(if (isNew) "Nouveau compte" else "Modifier le compte") },
        text = {
            Column(
                modifier = Modifier.verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                OutlinedTextField(
                    value = name, onValueChange = { name = it },
                    label = { Text("Nom (ex. M. Elasri – Informatique)") }, singleLine = true,
                    keyboardOptions = KeyboardOptions(capitalization = KeyboardCapitalization.Words)
                )
                if (isNew) {
                    OutlinedTextField(
                        value = email, onValueChange = { email = it.trim() },
                        label = { Text("E-mail") }, singleLine = true,
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Email)
                    )
                    OutlinedTextField(
                        value = password, onValueChange = { password = it },
                        label = { Text("Mot de passe provisoire") }, singleLine = true,
                        supportingText = { Text("À communiquer à l'enseignant (6 caractères min.).") }
                    )
                } else {
                    Text(email, style = MaterialTheme.typography.bodyMedium)
                }

                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Role.entries.forEach { r ->
                        FilterChip(
                            selected = role == r,
                            onClick = { if (!isSelf) role = r },
                            label = { Text(r.label) }
                        )
                    }
                }
                if (!isNew && !isSelf) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("Compte actif", modifier = Modifier.weight(1f))
                        Switch(checked = active, onCheckedChange = { active = it })
                    }
                }

                if (role == Role.TEACHER) {
                    Text("Classes attribuées", style = MaterialTheme.typography.titleSmall)
                    if (classes.isEmpty()) {
                        Text(
                            "Aucune classe : importez d'abord les listes Massar.",
                            style = MaterialTheme.typography.bodySmall
                        )
                    }
                    Column(Modifier.heightIn(max = 280.dp).verticalScroll(rememberScrollState())) {
                        classes.forEach { c ->
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                modifier = Modifier.fillMaxWidth().clickable {
                                    selected = if (c.id in selected) selected - c.id else selected + c.id
                                }
                            ) {
                                Checkbox(checked = c.id in selected, onCheckedChange = null)
                                Text(c.name, modifier = Modifier.padding(start = 8.dp))
                            }
                        }
                    }
                }
            }
        },
        confirmButton = {
            TextButton(enabled = valid, onClick = {
                onSave(name.trim(), email.trim(), password, role, if (role == Role.TEACHER) selected.toList() else emptyList(), active)
            }) { Text(if (isNew) "Créer" else "Enregistrer") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Annuler") } }
    )
}
