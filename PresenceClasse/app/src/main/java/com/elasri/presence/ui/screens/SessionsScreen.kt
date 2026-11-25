package com.elasri.presence.ui.screens

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CloudOff
import androidx.compose.material.icons.filled.DateRange
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Print
import androidx.compose.material.icons.filled.Share
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.elasri.presence.data.Session
import com.elasri.presence.data.frMessage
import com.elasri.presence.ui.SessionsViewModel
import com.elasri.presence.ui.factoryOf
import com.elasri.presence.ui.repository
import com.elasri.presence.ui.theme.AbsentColor
import com.elasri.presence.ui.theme.LateColor
import com.elasri.presence.util.fmtLong
import java.time.LocalDate

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SessionsScreen(classId: String, onBack: () -> Unit, onOpen: (String) -> Unit) {
    val repo = repository()
    val vm: SessionsViewModel = viewModel(factory = factoryOf { SessionsViewModel(repo, classId) })
    val result by vm.sessions.collectAsStateWithLifecycle()
    val cls by vm.schoolClass.collectAsStateWithLifecycle()
    var toDelete by remember { mutableStateOf<Session?>(null) }
    val sender = rememberAdminSender()
    val printer = rememberAdminPrinter()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Historique · ${cls?.name.orEmpty()}") },
                navigationIcon = { BackButton(onBack) }
            )
        }
    ) { pad ->
        val r = result
        val sessions = r?.getOrNull()
        when {
            r == null -> Box(Modifier.padding(pad).fillMaxSize(), contentAlignment = Alignment.Center) {
                CircularProgressIndicator()
            }
            sessions == null -> EmptyState(
                icon = Icons.Filled.Warning,
                title = "Historique indisponible",
                subtitle = r.exceptionOrNull()?.frMessage().orEmpty(),
                modifier = Modifier.padding(pad)
            )
            sessions.isEmpty() -> EmptyState(
                icon = Icons.Filled.DateRange,
                title = "Aucun appel",
                subtitle = "Les appels des 4 derniers mois apparaîtront ici, quel que soit l'enseignant.",
                modifier = Modifier.padding(pad)
            )
            else -> LazyColumn(Modifier.padding(pad).fillMaxSize()) {
                items(sessions, key = { it.id }) { s ->
                    val mine = vm.canEdit(s)
                    ListItem(
                        modifier = Modifier.clickable { onOpen(s.id) },
                        overlineContent = {
                            Text(listOf(s.period, s.teacherName).filter { it.isNotBlank() }.joinToString(" · "))
                        },
                        headlineContent = { Text(LocalDate.ofEpochDay(s.epochDay).fmtLong()) },
                        supportingContent = {
                            Text(
                                buildAnnotatedString {
                                    withStyle(SpanStyle(color = AbsentColor)) { append("${s.absents} absent(s)") }
                                    append("  ·  ")
                                    withStyle(SpanStyle(color = LateColor)) { append("${s.lates} retard(s)") }
                                    if (s.pending) append("  ·  en attente d'envoi")
                                },
                                style = MaterialTheme.typography.bodyMedium
                            )
                        },
                        trailingContent = {
                            Row {
                                if (s.pending) {
                                    Icon(
                                        Icons.Filled.CloudOff, contentDescription = "En attente d'envoi",
                                        tint = LateColor, modifier = Modifier.padding(12.dp)
                                    )
                                }
                                IconButton(onClick = { printer.print { repo.adminReportForSession(s.id) } }) {
                                    Icon(Icons.Filled.Print, contentDescription = "Imprimer")
                                }
                                IconButton(onClick = { sender.send { repo.adminReportForSession(s.id) } }) {
                                    Icon(Icons.Filled.Share, contentDescription = "Envoyer à l'administration")
                                }
                                if (mine) {
                                    IconButton(onClick = { toDelete = s }) {
                                        Icon(Icons.Filled.Delete, contentDescription = "Supprimer")
                                    }
                                }
                            }
                        }
                    )
                    HorizontalDivider()
                }
            }
        }
    }

    toDelete?.let { s ->
        ConfirmDialog(
            title = "Supprimer cet appel ?",
            text = LocalDate.ofEpochDay(s.epochDay).fmtLong() + (if (s.period.isNotBlank()) " (${s.period})" else ""),
            onConfirm = { vm.delete(s.id) },
            onDismiss = { toDelete = null }
        )
    }
}
