package com.etablissement.navette.ui.screens

import android.widget.Toast
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.DirectionsCar
import androidx.compose.material3.Checkbox
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.etablissement.navette.data.model.Group
import com.etablissement.navette.data.model.GroupPeriod
import com.etablissement.navette.ui.NavetteViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CreateEditGroupScreen(
    viewModel: NavetteViewModel,
    groupId: String?,
    onDone: () -> Unit
) {
    val context = LocalContext.current
    val teachers by viewModel.teachers.collectAsStateWithLifecycle()
    val myProfile by viewModel.myProfile.collectAsStateWithLifecycle()
    val myUid = viewModel.currentUid

    val existing = remember(groupId) { viewModel.groupById(groupId) }
    val isEdit = existing != null

    var name by remember { mutableStateOf(existing?.name ?: "") }
    var nameError by remember { mutableStateOf(false) }
    var period by remember { mutableStateOf(GroupPeriod.fromName(existing?.period)) }

    val selectedMemberIds = remember {
        mutableStateListOf<String>().apply { existing?.let { addAll(it.memberIds) } }
    }

    fun toggleMember(id: String) {
        if (selectedMemberIds.contains(id)) selectedMemberIds.remove(id)
        else selectedMemberIds.add(id)
    }

    fun save() {
        if (name.isBlank()) {
            nameError = true
            return
        }
        val driverId = existing?.driverId ?: myUid.orEmpty()
        val group = Group(
            id = existing?.id ?: "",
            name = name.trim(),
            driverId = driverId,
            driverName = myProfile?.name ?: existing?.driverName ?: "",
            period = period.name,
            memberIds = selectedMemberIds.toList()
        )
        viewModel.saveGroup(group)
        Toast.makeText(context, "Groupe enregistré", Toast.LENGTH_SHORT).show()
        onDone()
    }

    // Les membres sélectionnables = tous les enseignants sauf le conducteur (moi).
    val selectableTeachers = teachers.filter { it.id != (existing?.driverId ?: myUid) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(if (isEdit) "Modifier le groupe" else "Nouveau groupe") },
                navigationIcon = {
                    IconButton(onClick = onDone) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Retour")
                    }
                },
                actions = {
                    IconButton(onClick = { save() }) {
                        Icon(Icons.Filled.Check, contentDescription = "Enregistrer")
                    }
                }
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            OutlinedTextField(
                value = name,
                onValueChange = { name = it; nameError = false },
                label = { Text("Nom du groupe *") },
                isError = nameError,
                supportingText = if (nameError) {
                    { Text("Le nom est obligatoire") }
                } else null,
                singleLine = true,
                modifier = Modifier.fillMaxWidth()
            )

            Text(
                text = "Période de la navette",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Bold
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                GroupPeriod.entries.forEach { p ->
                    FilterChip(
                        selected = period == p,
                        onClick = { period = p },
                        label = { Text(p.label) }
                    )
                }
            }

            Spacer(Modifier.height(4.dp))

            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Filled.DirectionsCar, contentDescription = null)
                Spacer(Modifier.width(8.dp))
                Text(
                    text = "Conducteur : ${myProfile?.name ?: existing?.driverName ?: "moi"}",
                    style = MaterialTheme.typography.bodyMedium
                )
            }

            Text(
                text = "Membres (passagers)",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Bold
            )
            Text(
                text = "Cochez les enseignants à ajouter au groupe.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            if (selectableTeachers.isEmpty()) {
                Text(
                    text = "Aucun autre enseignant enregistré pour l'instant.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            } else {
                selectableTeachers.forEach { teacher ->
                    val checked = selectedMemberIds.contains(teacher.id)
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable { toggleMember(teacher.id) }
                            .padding(vertical = 4.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Checkbox(
                            checked = checked,
                            onCheckedChange = { toggleMember(teacher.id) }
                        )
                        Spacer(Modifier.width(8.dp))
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = teacher.name.ifBlank { "(sans nom)" },
                                style = MaterialTheme.typography.bodyLarge
                            )
                            if (teacher.subject.isNotBlank()) {
                                Text(
                                    text = teacher.subject,
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                        if (teacher.hasCar) {
                            Icon(
                                Icons.Filled.DirectionsCar,
                                contentDescription = "A une voiture",
                                tint = MaterialTheme.colorScheme.primary
                            )
                        }
                    }
                }
            }

            Spacer(Modifier.height(24.dp))
        }
    }
}
