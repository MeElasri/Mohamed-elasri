package com.etablissement.navette.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.DirectionsBus
import androidx.compose.material.icons.filled.NightsStay
import androidx.compose.material.icons.filled.Phone
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.WbSunny
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
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
fun HomeScreen(viewModel: NavetteViewModel) {
    val teachers by viewModel.teachers.collectAsStateWithLifecycle()
    val homeState by viewModel.homeState.collectAsStateWithLifecycle()

    val onNavette = viewModel.teachersOnNavette(teachers, homeState.day, homeState.period)

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
            // Bandeau d'information : jour + période en cours
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer
                )
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
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
                                text = if (homeState.isAuto)
                                    "Déterminée automatiquement selon l'heure"
                                else
                                    "Sélection manuelle",
                                style = MaterialTheme.typography.bodySmall
                            )
                        }
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

            // Sélecteur de jour (consultation)
            DaySelector(
                selectedDay = homeState.day,
                onDaySelected = { viewModel.selectDay(it) }
            )

            Spacer(Modifier.height(16.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Professeurs en navette",
                    style = MaterialTheme.typography.titleMedium
                )
                Text(
                    text = "${onNavette.size}",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary
                )
            }

            Spacer(Modifier.height(8.dp))

            if (onNavette.isEmpty()) {
                EmptyState(
                    message = "Aucun professeur prévu pour cette navette.\n" +
                        "Ajoutez des professeurs et leur planning dans l'onglet « Professeurs »."
                )
            } else {
                LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    items(onNavette, key = { it.id }) { teacher ->
                        TeacherNavetteCard(teacher)
                    }
                }
            }
        }
    }
}

@Composable
private fun TeacherNavetteCard(teacher: Teacher) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(Icons.Filled.DirectionsBus, contentDescription = null)
            Spacer(Modifier.width(16.dp))
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
            }
            if (teacher.phone.isNotBlank()) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        Icons.Filled.Phone,
                        contentDescription = "Téléphone",
                        modifier = Modifier.width(18.dp)
                    )
                    Spacer(Modifier.width(4.dp))
                    Text(teacher.phone, style = MaterialTheme.typography.bodySmall)
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun DaySelector(selectedDay: Weekday, onDaySelected: (Weekday) -> Unit) {
    LazyRowChips(
        items = Weekday.entries,
        isSelected = { it == selectedDay },
        label = { it.label.take(3) },
        onClick = onDaySelected
    )
}
