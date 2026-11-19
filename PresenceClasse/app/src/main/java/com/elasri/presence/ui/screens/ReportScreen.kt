package com.elasri.presence.ui.screens

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.List
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.AssistChip
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.elasri.presence.data.StudentReport
import com.elasri.presence.ui.ReportViewModel
import com.elasri.presence.ui.factoryOf
import com.elasri.presence.ui.rateText
import com.elasri.presence.ui.repository
import com.elasri.presence.ui.theme.AbsentColor
import com.elasri.presence.ui.theme.LateColor
import com.elasri.presence.util.fmtShort
import com.elasri.presence.util.schoolYearStart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.time.LocalDate

private enum class ExportKind { SUMMARY, DETAIL }

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ReportScreen(classId: String, onBack: () -> Unit) {
    val repo = repository()
    val vm: ReportViewModel = viewModel(factory = factoryOf { ReportViewModel(repo, classId) })
    val ui by vm.ui.collectAsStateWithLifecycle()

    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val snackbar = remember { SnackbarHostState() }

    var pickFrom by remember { mutableStateOf(false) }
    var pickTo by remember { mutableStateOf(false) }
    var menuOpen by remember { mutableStateOf(false) }
    var pendingExport by remember { mutableStateOf(ExportKind.SUMMARY) }

    val exportLauncher = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("text/csv")) { uri: Uri? ->
        if (uri != null) scope.launch {
            val msg = try {
                val bytes = if (pendingExport == ExportKind.SUMMARY) vm.summaryCsv() else vm.detailCsv()
                withContext(Dispatchers.IO) {
                    context.contentResolver.openOutputStream(uri)?.use { it.write(bytes) }
                        ?: error("fichier inaccessible")
                }
                "Fichier exporté."
            } catch (e: Exception) {
                "Export impossible : ${e.message}"
            }
            snackbar.showSnackbar(msg)
        }
    }

    fun fileName(kind: ExportKind): String {
        val base = ui.className.replace(Regex("[^A-Za-z0-9_-]+"), "_").ifBlank { "classe" }
        val prefix = if (kind == ExportKind.SUMMARY) "rapport" else "absences"
        return "${prefix}_${base}_${ui.from}_${ui.to}.csv"
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Rapport · ${ui.className}") },
                navigationIcon = { BackButton(onBack) },
                actions = {
                    IconButton(onClick = { menuOpen = true }, enabled = ui.rows.isNotEmpty()) {
                        Icon(Icons.Filled.Share, contentDescription = "Exporter")
                    }
                    DropdownMenu(expanded = menuOpen, onDismissRequest = { menuOpen = false }) {
                        DropdownMenuItem(
                            text = { Text("Exporter le résumé (CSV)") },
                            onClick = {
                                menuOpen = false
                                pendingExport = ExportKind.SUMMARY
                                exportLauncher.launch(fileName(ExportKind.SUMMARY))
                            }
                        )
                        DropdownMenuItem(
                            text = { Text("Exporter la liste des absences (CSV)") },
                            onClick = {
                                menuOpen = false
                                pendingExport = ExportKind.DETAIL
                                exportLauncher.launch(fileName(ExportKind.DETAIL))
                            }
                        )
                    }
                }
            )
        },
        snackbarHost = { SnackbarHost(snackbar) }
    ) { pad ->
        LazyColumn(
            modifier = Modifier.padding(pad).fillMaxSize(),
            contentPadding = PaddingValues(bottom = 24.dp)
        ) {
            item {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedButton(onClick = { pickFrom = true }, modifier = Modifier.weight(1f)) {
                            Text("Du ${ui.from.fmtShort()}")
                        }
                        OutlinedButton(onClick = { pickTo = true }, modifier = Modifier.weight(1f)) {
                            Text("Au ${ui.to.fmtShort()}")
                        }
                    }
                    Row(
                        modifier = Modifier.horizontalScroll(rememberScrollState()),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        val today = LocalDate.now()
                        AssistChip(onClick = { vm.setRange(today.withDayOfMonth(1), today) }, label = { Text("Ce mois") })
                        AssistChip(onClick = { vm.setRange(today.minusDays(29), today) }, label = { Text("30 derniers jours") })
                        AssistChip(onClick = { vm.setRange(today.minusDays(6), today) }, label = { Text("7 derniers jours") })
                        AssistChip(onClick = { vm.setRange(schoolYearStart(), today) }, label = { Text("Année scolaire") })
                    }
                    ElevatedCard(Modifier.fillMaxWidth()) {
                        Row(
                            modifier = Modifier.fillMaxWidth().padding(16.dp),
                            horizontalArrangement = Arrangement.SpaceAround
                        ) {
                            Stat("Séances", ui.totalSessions.toString(), MaterialTheme.colorScheme.primary)
                            Stat("Absences", ui.totalAbsences.toString(), AbsentColor)
                            Stat("Retards", ui.totalLates.toString(), LateColor)
                        }
                    }
                }
                HorizontalDivider()
            }

            when {
                ui.loading -> item {
                    Box(Modifier.fillMaxWidth().padding(32.dp), contentAlignment = Alignment.Center) {
                        CircularProgressIndicator()
                    }
                }
                ui.error != null -> item {
                    EmptyState(
                        icon = Icons.AutoMirrored.Filled.List,
                        title = "Rapport indisponible",
                        subtitle = ui.error.orEmpty()
                    )
                }
                ui.rows.isEmpty() -> item {
                    EmptyState(
                        icon = Icons.AutoMirrored.Filled.List,
                        title = "Pas de données",
                        subtitle = "Ajoutez des élèves et faites l'appel pour voir le rapport."
                    )
                }
                else -> items(ui.rows, key = { it.studentId }) { r ->
                    ReportRow(r)
                    HorizontalDivider()
                }
            }
        }
    }

    if (pickFrom) DatePickDialog(ui.from, onDismiss = { pickFrom = false }) { vm.setRange(it, ui.to) }
    if (pickTo) DatePickDialog(ui.to, onDismiss = { pickTo = false }) { vm.setRange(ui.from, it) }
}

@Composable
private fun Stat(label: String, value: String, color: Color) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(value, style = MaterialTheme.typography.headlineMedium, color = color, fontWeight = FontWeight.Bold)
        Text(label, style = MaterialTheme.typography.bodySmall)
    }
}

@Composable
private fun ReportRow(r: StudentReport) {
    val alert = r.rate >= 0.2f
    Column(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 10.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text(r.name, style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Medium)
                Text(
                    "${r.absences} abs. · ${r.lates} ret. · ${r.recorded} séance(s)" +
                        if (r.massarCode.isNotBlank()) " · ${r.massarCode}" else "",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            Text(
                "${r.rateText()} %",
                style = MaterialTheme.typography.titleMedium,
                color = if (alert) AbsentColor else MaterialTheme.colorScheme.onSurface,
                fontWeight = FontWeight.Bold
            )
        }
        Spacer(Modifier.height(6.dp))
        LinearProgressIndicator(
            progress = { r.rate.coerceIn(0f, 1f) },
            modifier = Modifier.fillMaxWidth(),
            color = if (alert) AbsentColor else MaterialTheme.colorScheme.primary
        )
    }
}
