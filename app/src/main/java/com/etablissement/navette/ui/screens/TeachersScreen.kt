package com.etablissement.navette.ui.screens

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.etablissement.navette.data.model.Period
import com.etablissement.navette.data.model.Teacher
import com.etablissement.navette.data.model.Weekday
import com.etablissement.navette.ui.NavetteViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TeachersScreen(
    viewModel: NavetteViewModel,
    onAddTeacher: () -> Unit,
    onEditTeacher: (String) -> Unit
) {
    val teachers by viewModel.teachers.collectAsStateWithLifecycle()
    var teacherToDelete by remember { mutableStateOf<Teacher?>(null) }

    Scaffold(
        topBar = { TopAppBar(title = { Text("Professeurs") }) },
        floatingActionButton = {
            FloatingActionButton(onClick = onAddTeacher) {
                Icon(Icons.Filled.Add, contentDescription = "Ajouter un professeur")
            }
        }
    ) { padding ->
        if (teachers.isEmpty()) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding)
            ) {
                EmptyState(
                    message = "Aucun professeur enregistré.\n" +
                        "Appuyez sur + pour ajouter un professeur et son planning de navette."
                )
            }
        } else {
            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding)
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                items(teachers, key = { it.id }) { teacher ->
                    TeacherRow(
                        teacher = teacher,
                        onEdit = { onEditTeacher(teacher.id) },
                        onDelete = { teacherToDelete = teacher }
                    )
                }
            }
        }
    }

    teacherToDelete?.let { teacher ->
        AlertDialog(
            onDismissRequest = { teacherToDelete = null },
            title = { Text("Supprimer ?") },
            text = { Text("Supprimer définitivement « ${teacher.name} » ?") },
            confirmButton = {
                TextButton(onClick = {
                    viewModel.deleteTeacher(teacher.id)
                    teacherToDelete = null
                }) { Text("Supprimer") }
            },
            dismissButton = {
                TextButton(onClick = { teacherToDelete = null }) { Text("Annuler") }
            }
        )
    }
}

@Composable
private fun TeacherRow(
    teacher: Teacher,
    onEdit: () -> Unit,
    onDelete: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onEdit)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = teacher.name,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold
                )
                if (teacher.subject.isNotBlank()) {
                    Text(
                        text = teacher.subject,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                Spacer(Modifier.width(4.dp))
                Text(
                    text = scheduleSummary(teacher),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.primary
                )
            }
            IconButton(onClick = onEdit) {
                Icon(Icons.Filled.Edit, contentDescription = "Modifier")
            }
            IconButton(onClick = onDelete) {
                Icon(Icons.Filled.Delete, contentDescription = "Supprimer")
            }
        }
    }
}

/** Résumé lisible du planning, ex : "Lun (Matin), Mar (Matin, Soir)". */
private fun scheduleSummary(teacher: Teacher): String {
    val parts = Weekday.entries.mapNotNull { day ->
        val periods = Period.entries.filter { teacher.isScheduled(day, it) }
        if (periods.isEmpty()) null
        else "${day.label.take(3)} (${periods.joinToString(", ") { it.label }})"
    }
    return if (parts.isEmpty()) "Aucune navette planifiée" else parts.joinToString(" · ")
}
