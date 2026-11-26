package com.elasri.presence.ui.screens

import android.content.Context
import androidx.compose.foundation.background
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
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowLeft
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowRight
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.FamilyRestroom
import androidx.compose.material.icons.filled.Print
import androidx.compose.material.icons.filled.Remove
import androidx.compose.material.icons.filled.Share
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedIconButton
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.elasri.presence.data.WeeklyStats
import com.elasri.presence.ui.StatsViewModel
import com.elasri.presence.ui.factoryOf
import com.elasri.presence.ui.repository
import com.elasri.presence.ui.theme.AbsentColor
import com.elasri.presence.ui.theme.LateColor
import com.elasri.presence.util.fmtShort
import java.util.Locale

private enum class StatsTab(val label: String) { CONVOKE("À convoquer"), STUDENTS("Tous les élèves"), CLASSES("Par classe") }

/** Administration: absences of the week and the students whose parents must be called in. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun StatsScreen(onBack: () -> Unit) {
    val repo = repository()
    val context = LocalContext.current
    val vm: StatsViewModel = viewModel(factory = factoryOf {
        StatsViewModel(repo, context.applicationContext.getSharedPreferences("settings", Context.MODE_PRIVATE))
    })
    val ui by vm.ui.collectAsStateWithLifecycle()
    val sender = rememberAdminSender()
    val printer = rememberAdminPrinter()
    var tab by remember { mutableStateOf(StatsTab.CONVOKE) }
    var pickWeek by remember { mutableStateOf(false) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Statistiques de la semaine") },
                navigationIcon = { BackButton(onBack) },
                actions = {
                    IconButton(
                        onClick = { vm.report()?.let { out -> printer.print { out } } },
                        enabled = ui.result != null && !ui.loading
                    ) {
                        Icon(Icons.Filled.Print, contentDescription = "Imprimer")
                    }
                    IconButton(
                        onClick = { vm.report()?.let { out -> sender.send { out } } },
                        enabled = ui.result != null && !ui.loading
                    ) {
                        Icon(Icons.Filled.Share, contentDescription = "Envoyer (Excel)")
                    }
                }
            )
        }
    ) { pad ->
        LazyColumn(
            modifier = Modifier.padding(pad).fillMaxSize(),
            contentPadding = PaddingValues(bottom = 24.dp)
        ) {
            // ---- week selector ----
            item {
                Row(
                    modifier = Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 4.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    IconButton(onClick = vm::previousWeek) {
                        Icon(Icons.AutoMirrored.Filled.KeyboardArrowLeft, contentDescription = "Semaine précédente")
                    }
                    TextButton(onClick = { pickWeek = true }, modifier = Modifier.weight(1f)) {
                        Text("Du ${ui.weekStart.fmtShort()} au ${ui.weekEnd.fmtShort()}", fontWeight = FontWeight.SemiBold)
                    }
                    IconButton(onClick = vm::nextWeek) {
                        Icon(Icons.AutoMirrored.Filled.KeyboardArrowRight, contentDescription = "Semaine suivante")
                    }
                }
            }

            // ---- totals ----
            item {
                ElevatedCard(Modifier.fillMaxWidth().padding(horizontal = 16.dp)) {
                    Row(Modifier.fillMaxWidth().padding(16.dp), horizontalArrangement = Arrangement.SpaceAround) {
                        BigStat("Appels", ui.result?.sessions, MaterialTheme.colorScheme.primary)
                        BigStat("Absences", ui.result?.absences, AbsentColor)
                        BigStat("Retards", ui.result?.lates, LateColor)
                        BigStat("À convoquer", if (ui.result == null) null else ui.convoke.size, Color(0xFF6A1B9A))
                    }
                }
            }

            // ---- threshold ----
            item {
                Row(
                    modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 12.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column(Modifier.weight(1f)) {
                        Text("Seuil de convocation", style = MaterialTheme.typography.titleSmall)
                        Text(
                            "Parents convoqués à partir de ${ui.threshold} absence(s) dans la semaine",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                    OutlinedIconButton(onClick = { vm.setThreshold(ui.threshold - 1) }, enabled = ui.threshold > 1) {
                        Icon(Icons.Filled.Remove, contentDescription = "Diminuer")
                    }
                    Text("${ui.threshold}", style = MaterialTheme.typography.titleLarge, modifier = Modifier.padding(horizontal = 12.dp))
                    OutlinedIconButton(onClick = { vm.setThreshold(ui.threshold + 1) }) {
                        Icon(Icons.Filled.Add, contentDescription = "Augmenter")
                    }
                }
                HorizontalDivider()
            }

            // ---- tabs ----
            item {
                Row(Modifier.fillMaxWidth().padding(horizontal = 8.dp)) {
                    StatsTab.entries.forEach { t ->
                        TextButton(onClick = { tab = t }, modifier = Modifier.weight(1f)) {
                            Text(
                                t.label,
                                fontWeight = if (tab == t) FontWeight.Bold else FontWeight.Normal,
                                color = if (tab == t) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }
                HorizontalDivider()
            }

            val r = ui.result
            when {
                ui.loading -> item {
                    Box(Modifier.fillMaxWidth().padding(32.dp), contentAlignment = Alignment.Center) { CircularProgressIndicator() }
                }
                ui.error != null -> item {
                    EmptyState(Icons.Filled.Warning, "Statistiques indisponibles", ui.error.orEmpty())
                }
                r == null || r.sessions == 0 -> item {
                    EmptyState(Icons.Filled.Warning, "Aucun appel cette semaine", "Choisissez une autre semaine avec les flèches.")
                }
                tab == StatsTab.CONVOKE -> {
                    if (ui.convoke.isEmpty()) item {
                        EmptyState(
                            Icons.Filled.FamilyRestroom, "Aucune convocation",
                            "Aucun élève n'a ${ui.threshold} absence(s) ou plus cette semaine."
                        )
                    }
                    items(ui.convoke, key = { "c_${it.classId}_${it.studentId}" }) { s -> StudentRow(s, highlight = true) }
                }
                tab == StatsTab.STUDENTS ->
                    r.students
                        .groupBy { it.classId }
                        .values
                        .sortedBy { it.first().className.lowercase() }
                        .forEach { studs ->
                            val cls = studs.first()
                            item(key = "h_${cls.classId}") { ClassHeader(cls.className, studs) }
                            items(studs, key = { "s_${it.classId}_${it.studentId}" }) { s ->
                                StudentRow(s, highlight = s.absences >= ui.threshold, showClass = false)
                            }
                        }
                else -> items(r.classes, key = { "k_${it.classId}" }) { c -> ClassRow(c, r.absences) }
            }
        }
    }

    if (pickWeek) {
        DatePickDialog(initial = ui.weekStart, onDismiss = { pickWeek = false }, onPick = vm::setWeek)
    }
}

@Composable
private fun BigStat(label: String, value: Int?, color: Color) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(value?.toString() ?: "–", style = MaterialTheme.typography.headlineSmall, color = color, fontWeight = FontWeight.Bold)
        Text(label, style = MaterialTheme.typography.bodySmall)
    }
}

private fun hoursText(h: Double?): String =
    if (h == null) "" else " · " + String.format(Locale.FRENCH, "%.1f", h).removeSuffix(",0") + " h"

/** Section header for the "Tous les élèves" tab, grouped by class. */
@Composable
private fun ClassHeader(className: String, students: List<WeeklyStats.StudentWeek>) {
    Row(
        Modifier
            .fillMaxWidth()
            .background(MaterialTheme.colorScheme.secondaryContainer)
            .padding(horizontal = 16.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            className,
            style = MaterialTheme.typography.titleSmall,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onSecondaryContainer,
            modifier = Modifier.weight(1f)
        )
        Text(
            "${students.size} élève(s) · ${students.sumOf { it.absences }} abs.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSecondaryContainer
        )
    }
}

@Composable
private fun StudentRow(s: WeeklyStats.StudentWeek, highlight: Boolean, showClass: Boolean = true) {
    Column(
        Modifier
            .fillMaxWidth()
            .background(if (highlight) AbsentColor.copy(alpha = 0.06f) else Color.Transparent)
            .padding(horizontal = 16.dp, vertical = 10.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text(s.name, style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Medium)
                val sub = listOf(if (showClass) s.className else "", s.massarCode).filter { it.isNotBlank() }.joinToString(" · ")
                if (sub.isNotBlank()) {
                    Text(
                        sub,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
            Box(
                Modifier
                    .background(if (highlight) AbsentColor else MaterialTheme.colorScheme.surfaceVariant, RoundedCornerShape(12.dp))
                    .padding(horizontal = 10.dp, vertical = 4.dp)
            ) {
                Text(
                    "${s.absences} abs.",
                    color = if (highlight) Color.White else MaterialTheme.colorScheme.onSurfaceVariant,
                    fontWeight = FontWeight.Bold
                )
            }
        }
        val details = s.absenceDetails()
        Text(
            (if (details.isBlank()) "" else details) + hoursText(s.absentHours) +
                if (s.lates > 0) " · ${s.lates} retard(s)" else "",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 2.dp)
        )
    }
    HorizontalDivider()
}

@Composable
private fun ClassRow(c: WeeklyStats.ClassWeek, totalAbsences: Int) {
    Column(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 10.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(c.className, style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Medium, modifier = Modifier.weight(1f))
            Text("${c.absences} abs.", color = AbsentColor, fontWeight = FontWeight.Bold)
        }
        Text(
            "${c.sessions} appel(s) · ${c.lates} retard(s) · ${c.studentsAbsent} élève(s) absent(s)",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Spacer(Modifier.size(6.dp))
        val frac = if (totalAbsences == 0) 0f else c.absences.toFloat() / totalAbsences
        LinearProgressIndicator(
            progress = { frac.coerceIn(0f, 1f) },
            color = AbsentColor,
            modifier = Modifier.fillMaxWidth()
        )
    }
    HorizontalDivider()
}
