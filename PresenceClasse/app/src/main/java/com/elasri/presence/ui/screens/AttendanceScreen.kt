package com.elasri.presence.ui.screens

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.CloudOff
import androidx.compose.material.icons.filled.DateRange
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Share
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.BottomAppBar
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
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
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.launch
import com.elasri.presence.data.Session
import com.elasri.presence.data.Status
import com.elasri.presence.ui.AttendanceEvent
import com.elasri.presence.ui.AttendanceViewModel
import com.elasri.presence.ui.factoryOf
import com.elasri.presence.ui.repository
import com.elasri.presence.ui.theme.AbsentColor
import com.elasri.presence.ui.theme.LateColor
import com.elasri.presence.ui.theme.PresentColor
import com.elasri.presence.util.fmtLong
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter

private val PeriodSuggestions = listOf("8h-10h", "10h-12h", "14h-16h", "16h-18h")
private val ReturnedColor = Color(0xFFEF6C00)
private val timeFmt = DateTimeFormatter.ofPattern("HH:mm")

fun Status.color(): Color = when (this) {
    Status.PRESENT -> PresentColor
    Status.ABSENT -> AbsentColor
    Status.LATE -> LateColor
}

private fun Long.hhmm(): String =
    if (this <= 0) "" else Instant.ofEpochMilli(this).atZone(ZoneId.systemDefault()).format(timeFmt)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AttendanceScreen(classId: String, sessionId: String?, onBack: () -> Unit) {
    val repo = repository()
    val vm: AttendanceViewModel = viewModel(factory = factoryOf { AttendanceViewModel(repo, classId, sessionId) })
    val ui by vm.ui.collectAsStateWithLifecycle()
    val snackbar = remember { SnackbarHostState() }
    val sender = rememberAdminSender()

    var pickDate by remember { mutableStateOf(false) }
    var confirmLeave by remember { mutableStateOf(false) }
    var saved by remember { mutableStateOf<AttendanceEvent.Saved?>(null) }

    LaunchedEffect(Unit) {
        vm.events.collect { ev ->
            when (ev) {
                is AttendanceEvent.Message -> launch { snackbar.showSnackbar(ev.text) }
                is AttendanceEvent.Saved -> saved = ev
            }
        }
    }

    val tryLeave: () -> Unit = {
        if (ui.dirty && !ui.readOnly) {
            confirmLeave = true
        } else {
            onBack()
        }
    }
    BackHandler(enabled = ui.dirty && !ui.readOnly) { confirmLeave = true }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text(
                            when {
                                ui.readOnly -> "Appel (lecture seule)"
                                ui.editing != null -> "Modifier l'appel"
                                else -> "Faire l'appel"
                            }
                        )
                        Text(
                            ui.cls?.name.orEmpty(),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                },
                navigationIcon = { BackButton(tryLeave) }
            )
        },
        snackbarHost = { SnackbarHost(snackbar) },
        bottomBar = {
            if (!ui.loading && ui.error == null) {
                BottomAppBar {
                    Row(
                        modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text("${ui.count(Status.ABSENT)} absent(s)", color = AbsentColor, fontWeight = FontWeight.SemiBold)
                            Text(
                                "${ui.count(Status.LATE)} retard(s) · ${ui.count(Status.PRESENT)} présent(s)",
                                style = MaterialTheme.typography.bodySmall
                            )
                        }
                        if (ui.readOnly) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(Icons.Filled.Lock, contentDescription = null, modifier = Modifier.size(18.dp))
                                Spacer(Modifier.width(6.dp))
                                Text(ui.editing?.teacherName.orEmpty(), style = MaterialTheme.typography.bodySmall)
                            }
                        } else {
                            Button(onClick = vm::save, enabled = !ui.saving && ui.students.isNotEmpty()) {
                                if (ui.saving) {
                                    CircularProgressIndicator(strokeWidth = 2.dp, modifier = Modifier.size(18.dp), color = Color.White)
                                } else {
                                    Icon(Icons.Filled.Check, contentDescription = null)
                                }
                                Spacer(Modifier.width(8.dp))
                                Text("Enregistrer")
                            }
                        }
                    }
                }
            }
        }
    ) { pad ->
        when {
            ui.loading -> Box(Modifier.padding(pad).fillMaxSize(), contentAlignment = Alignment.Center) {
                CircularProgressIndicator()
            }
            ui.error != null -> EmptyState(
                icon = Icons.Filled.Warning,
                title = "Impossible d'ouvrir l'appel",
                subtitle = ui.error.orEmpty(),
                modifier = Modifier.padding(pad)
            )
            else -> LazyColumn(
                modifier = Modifier.padding(pad).fillMaxSize(),
                contentPadding = PaddingValues(bottom = 16.dp)
            ) {
                item {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        val canChangeSlot = !vm.isEditing && !ui.readOnly
                        OutlinedButton(
                            onClick = { pickDate = true },
                            enabled = canChangeSlot,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Icon(Icons.Filled.DateRange, contentDescription = null)
                            Spacer(Modifier.width(8.dp))
                            Text(ui.date.fmtLong())
                        }
                        OutlinedTextField(
                            value = ui.period,
                            onValueChange = vm::setPeriod,
                            enabled = canChangeSlot,
                            label = { Text("Séance") },
                            placeholder = { Text("ex. 8h-10h") },
                            singleLine = true,
                            modifier = Modifier.fillMaxWidth()
                        )
                        if (canChangeSlot) {
                            Row(
                                modifier = Modifier.horizontalScroll(rememberScrollState()),
                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                PeriodSuggestions.forEach { p ->
                                    FilterChip(
                                        selected = ui.period == p,
                                        onClick = { vm.setPeriod(if (ui.period == p) "" else p) },
                                        label = { Text(p) }
                                    )
                                }
                            }
                        }

                        PreviousLessonBanner(
                            previous = ui.previous,
                            isNew = ui.editing == null && !ui.readOnly,
                            onReapply = vm::applyPrevious
                        )

                        if (ui.returned.isNotEmpty()) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(Icons.Filled.Warning, contentDescription = null, tint = ReturnedColor, modifier = Modifier.size(18.dp))
                                Spacer(Modifier.width(6.dp))
                                Text(
                                    "${ui.returned.size} élève(s) absent(s) à la séance précédente sont là maintenant : billet d'entrée ?",
                                    color = ReturnedColor,
                                    style = MaterialTheme.typography.bodySmall
                                )
                            }
                        }

                        if (!ui.readOnly) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Text(
                                    "Touchez un élève pour le marquer absent.",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    modifier = Modifier.weight(1f)
                                )
                                AssistChip(
                                    onClick = vm::markAllPresent,
                                    label = { Text("Tous présents") },
                                    leadingIcon = { Icon(Icons.Filled.Refresh, contentDescription = null) }
                                )
                            }
                        }
                    }
                    HorizontalDivider()
                }

                itemsIndexed(ui.students, key = { _, s -> s.id }) { index, s ->
                    val status = ui.marks[s.id] ?: Status.PRESENT
                    val wasAbsent = ui.previous?.marks?.get(s.id) == Status.ABSENT
                    val returned = s.id in ui.returned
                    val bg = when {
                        status != Status.PRESENT -> status.color().copy(alpha = 0.10f)
                        returned -> ReturnedColor.copy(alpha = 0.08f)
                        else -> Color.Transparent
                    }
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(bg)
                            .clickable(enabled = !ui.readOnly) { vm.toggle(s.id) }
                            .padding(horizontal = 16.dp, vertical = 8.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            "${index + 1}",
                            style = MaterialTheme.typography.labelLarge,
                            color = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.widthIn(min = 28.dp)
                        )
                        Column(Modifier.weight(1f)) {
                            Text(
                                s.name,
                                style = MaterialTheme.typography.bodyLarge,
                                fontWeight = if (status == Status.PRESENT) FontWeight.Normal else FontWeight.SemiBold
                            )
                            val sub = listOfNotNull(
                                s.massarCode.takeIf { it.isNotBlank() },
                                if (wasAbsent) "absent à la séance précédente" else null
                            ).joinToString(" · ")
                            if (sub.isNotBlank()) {
                                Text(
                                    sub,
                                    style = MaterialTheme.typography.bodySmall,
                                    color = if (returned) ReturnedColor else MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                        Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                            Status.entries.forEach { st ->
                                FilterChip(
                                    selected = status == st,
                                    onClick = { vm.setStatus(s.id, st) },
                                    label = { Text(st.short, fontWeight = FontWeight.Bold) },
                                    colors = FilterChipDefaults.filterChipColors(
                                        selectedContainerColor = st.color(),
                                        selectedLabelColor = Color.White
                                    )
                                )
                            }
                        }
                    }
                    HorizontalDivider()
                }
            }
        }
    }

    if (pickDate) {
        DatePickDialog(initial = ui.date, onDismiss = { pickDate = false }, onPick = vm::setDate)
    }
    if (confirmLeave) {
        ConfirmDialog(
            title = "Quitter sans enregistrer ?",
            text = "Les modifications de cet appel seront perdues.",
            confirmLabel = "Quitter",
            onConfirm = onBack,
            onDismiss = { confirmLeave = false }
        )
    }

    saved?.let { ev ->
        AlertDialog(
            onDismissRequest = { saved = null; onBack() },
            icon = {
                Icon(
                    if (ev.confirmed) Icons.Filled.Check else Icons.Filled.CloudOff,
                    contentDescription = null,
                    tint = if (ev.confirmed) PresentColor else LateColor
                )
            },
            title = { Text(if (ev.confirmed) "Appel enregistré" else "Enregistré sur le téléphone") },
            text = {
                Text(
                    (if (ev.confirmed)
                        "Les collègues qui ont ensuite cette classe voient déjà ces absences.\n\n"
                    else
                        "Pas de connexion pour le moment : l'appel sera envoyé automatiquement dès le retour " +
                            "d'Internet. Les collègues le verront à ce moment-là.\n\n") +
                        "${ui.count(Status.ABSENT)} absent(s), ${ui.count(Status.LATE)} retard(s).\n" +
                        "Envoyer aussi la liste à l'administration (WhatsApp, e-mail…) ?"
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    saved = null
                    sender.send(after = onBack) { repo.adminReportForSession(ev.sessionId) }
                }) {
                    Icon(Icons.Filled.Share, contentDescription = null)
                    Spacer(Modifier.width(8.dp))
                    Text("Envoyer")
                }
            },
            dismissButton = { TextButton(onClick = { saved = null; onBack() }) { Text("Terminer") } }
        )
    }
}

@Composable
private fun PreviousLessonBanner(previous: Session?, isNew: Boolean, onReapply: () -> Unit) {
    val shape = RoundedCornerShape(12.dp)
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(MaterialTheme.colorScheme.secondaryContainer, shape)
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Filled.History, contentDescription = null, modifier = Modifier.size(18.dp))
            Spacer(Modifier.width(6.dp))
            Text("Séance précédente", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
        }
        if (previous == null) {
            Text(
                "Aucun autre appel aujourd'hui pour cette classe (ou pas encore reçu).",
                style = MaterialTheme.typography.bodySmall
            )
            return@Column
        }
        val header = listOf(previous.period, previous.teacherName).filter { it.isNotBlank() }.joinToString(" · ")
        val time = previous.createdAt.hhmm()
        Text(
            header + (if (time.isNotBlank()) " — enregistré à $time" else "") +
                if (previous.pending) " (en attente d'envoi)" else "",
            style = MaterialTheme.typography.bodySmall
        )
        val absents = previous.nonPresent.filter { it.status == Status.ABSENT }
        val lates = previous.nonPresent.filter { it.status == Status.LATE }
        Text(
            if (absents.isEmpty()) "Aucun absent."
            else "${absents.size} absent(s) : " + absents.joinToString(", ") { it.name },
            style = MaterialTheme.typography.bodyMedium,
            color = if (absents.isEmpty()) MaterialTheme.colorScheme.onSecondaryContainer else AbsentColor,
            fontWeight = FontWeight.Medium
        )
        if (lates.isNotEmpty()) {
            Text(
                "${lates.size} retard(s) : " + lates.joinToString(", ") { it.name },
                style = MaterialTheme.typography.bodySmall,
                color = LateColor
            )
        }
        if (isNew && absents.isNotEmpty()) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    "Ces élèves sont pré-marqués absents.",
                    style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.weight(1f)
                )
                TextButton(onClick = onReapply) { Text("Réappliquer") }
            }
        }
    }
}
