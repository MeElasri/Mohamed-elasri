package com.etablissement.navette.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Call
import androidx.compose.material.icons.filled.DirectionsCar
import androidx.compose.material.icons.filled.NightsStay
import androidx.compose.material.icons.filled.PersonOutline
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Sms
import androidx.compose.material.icons.filled.WbSunny
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalIconButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.etablissement.navette.R
import com.etablissement.navette.data.model.Period
import com.etablissement.navette.data.model.Teacher
import com.etablissement.navette.data.model.Weekday
import com.etablissement.navette.ui.NavetteViewModel
import com.etablissement.navette.util.ContactActions

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(viewModel: NavetteViewModel) {
    val teachers by viewModel.teachers.collectAsStateWithLifecycle()
    val homeState by viewModel.homeState.collectAsStateWithLifecycle()

    // Calculs mis en cache : recalculés seulement si la liste, le jour ou la période change.
    val onNavette = remember(teachers, homeState.day, homeState.period) {
        viewModel.teachersOnNavette(teachers, homeState.day, homeState.period)
    }
    val drivers = remember(onNavette) { onNavette.filter { it.hasCar } }
    val passengers = remember(onNavette) { onNavette.filterNot { it.hasCar } }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Navette des professeurs") },
                actions = {
                    IconButton(onClick = { viewModel.refreshNow() }) {
                        Icon(Icons.Filled.Refresh, contentDescription = "Actualiser maintenant")
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
        ) {
            // Bandeau : jour + période en cours
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer
                )
            ) {
                Row(
                    modifier = Modifier.padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = if (homeState.period == Period.MORNING)
                            Icons.Filled.WbSunny else Icons.Filled.NightsStay,
                        contentDescription = null
                    )
                    Spacer(Modifier.width(12.dp))
                    Column {
                        Text(
                            text = "${homeState.day.label} • Navette du ${homeState.period.label.lowercase()}",
                            style = MaterialTheme.typography.titleLarge,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = if (homeState.isAuto) "Déterminée automatiquement selon l'heure"
                            else "Sélection manuelle",
                            style = MaterialTheme.typography.bodySmall
                        )
                    }
                }
            }

            Spacer(Modifier.height(12.dp))

            // Sélecteurs Matin / Soir
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Period.entries.forEach { period ->
                    FilterChip(
                        selected = homeState.period == period,
                        onClick = { viewModel.selectPeriod(period) },
                        label = { Text(period.label) },
                        leadingIcon = {
                            Icon(
                                imageVector = if (period == Period.MORNING)
                                    Icons.Filled.WbSunny else Icons.Filled.NightsStay,
                                contentDescription = null
                            )
                        }
                    )
                }
            }

            Spacer(Modifier.height(8.dp))

            LazyRowChips(
                items = Weekday.entries,
                isSelected = { it == homeState.day },
                label = { it.label.take(3) },
                onClick = { viewModel.selectDay(it) }
            )

            Spacer(Modifier.height(16.dp))

            if (onNavette.isEmpty()) {
                EmptyState(
                    message = "Aucun professeur prévu pour cette navette.\n" +
                        "Renseignez votre emploi du temps dans « Mon profil »."
                )
            } else {
                LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    if (drivers.isNotEmpty()) {
                        item {
                            SectionHeader(
                                icon = Icons.Filled.DirectionsCar,
                                title = "Conducteurs (avec voiture)",
                                count = drivers.size
                            )
                        }
                        items(drivers, key = { it.id }) { TeacherNavetteCard(it, isDriver = true) }
                    }
                    if (passengers.isNotEmpty()) {
                        item {
                            SectionHeader(
                                icon = Icons.Filled.PersonOutline,
                                title = "Passagers (sans voiture)",
                                count = passengers.size
                            )
                        }
                        items(passengers, key = { it.id }) { TeacherNavetteCard(it, isDriver = false) }
                    }
                }
            }
        }
    }
}

@Composable
private fun SectionHeader(icon: ImageVector, title: String, count: Int) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(top = 8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(icon, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
        Spacer(Modifier.width(8.dp))
        Text(
            text = title,
            style = MaterialTheme.typography.titleSmall,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.weight(1f)
        )
        Text(
            text = "$count",
            style = MaterialTheme.typography.titleSmall,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.primary
        )
    }
}

@Composable
private fun TeacherNavetteCard(teacher: Teacher, isDriver: Boolean) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = if (isDriver) Icons.Filled.DirectionsCar else Icons.Filled.PersonOutline,
                contentDescription = null
            )
            Spacer(Modifier.width(16.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = teacher.name,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold
                )
                val sub = buildString {
                    if (teacher.subject.isNotBlank()) append(teacher.subject)
                    if (teacher.phone.isNotBlank()) {
                        if (isNotEmpty()) append(" • ")
                        append(teacher.phone)
                    }
                }
                if (sub.isNotBlank()) {
                    Text(
                        text = sub,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            // Actions de communication (Appeler / SMS / WhatsApp)
            ContactButtons(phone = teacher.phone)
        }
    }
}
