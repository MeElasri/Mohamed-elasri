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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Logout
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.DirectionsCar
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenu
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.text.KeyboardOptions
import android.widget.Toast
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.etablissement.navette.data.model.Period
import com.etablissement.navette.data.model.Subjects
import com.etablissement.navette.data.model.Teacher
import com.etablissement.navette.data.model.Weekday
import com.etablissement.navette.ui.NavetteViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ProfileScreen(viewModel: NavetteViewModel) {
    val context = LocalContext.current
    val profile by viewModel.myProfile.collectAsStateWithLifecycle()

    var name by remember { mutableStateOf("") }
    var subject by remember { mutableStateOf("") }
    var phone by remember { mutableStateOf("") }
    var hasCar by remember { mutableStateOf(false) }
    var nameError by remember { mutableStateOf(false) }
    var initialized by remember { mutableStateOf(false) }

    val selected = remember { mutableStateMapOf<Pair<Weekday, Period>, Boolean>() }

    // Pré-remplit le formulaire une fois le profil chargé depuis Firestore.
    LaunchedEffect(profile) {
        val p = profile
        if (p != null && !initialized) {
            name = p.name
            subject = p.subject
            phone = p.phone
            hasCar = p.hasCar
            selected.clear()
            Weekday.entries.forEach { day ->
                Period.entries.forEach { period ->
                    if (p.isScheduled(day, period)) selected[day to period] = true
                }
            }
            initialized = true
        }
    }

    fun toggle(day: Weekday, period: Period) {
        val key = day to period
        selected[key] = !(selected[key] ?: false)
    }

    fun buildSchedule(): Map<String, List<String>> {
        val result = mutableMapOf<String, MutableList<String>>()
        selected.filterValues { it }.keys.forEach { (day, period) ->
            result.getOrPut(day.name) { mutableListOf() }.add(period.name)
        }
        return result
    }

    fun save() {
        if (name.isBlank()) {
            nameError = true
            return
        }
        viewModel.saveMyProfile(
            Teacher(
                name = name.trim(),
                subject = subject.trim(),
                phone = phone.trim(),
                email = viewModel.currentEmail ?: "",
                hasCar = hasCar,
                schedule = buildSchedule()
            )
        )
        Toast.makeText(context, "Profil enregistré", Toast.LENGTH_SHORT).show()
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Mon profil") },
                actions = {
                    IconButton(onClick = { save() }) {
                        Icon(Icons.Filled.Check, contentDescription = "Enregistrer")
                    }
                    IconButton(onClick = { viewModel.signOut() }) {
                        Icon(Icons.AutoMirrored.Filled.Logout, contentDescription = "Se déconnecter")
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
            viewModel.currentEmail?.let {
                Text(
                    text = "Connecté : $it",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            OutlinedTextField(
                value = name,
                onValueChange = { name = it; nameError = false },
                label = { Text("Nom complet *") },
                isError = nameError,
                supportingText = if (nameError) {
                    { Text("Le nom est obligatoire") }
                } else null,
                singleLine = true,
                modifier = Modifier.fillMaxWidth()
            )

            SubjectDropdown(
                selected = subject,
                onSelected = { subject = it }
            )

            OutlinedTextField(
                value = phone,
                onValueChange = { phone = it },
                label = { Text("Téléphone (ex: +212 6 12 34 56 78)") },
                singleLine = true,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Phone),
                modifier = Modifier.fillMaxWidth()
            )

            // Possède une voiture ?
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(Icons.Filled.DirectionsCar, contentDescription = null)
                Spacer(Modifier.width(12.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text("Je possède une voiture", style = MaterialTheme.typography.bodyLarge)
                    Text(
                        text = "Je peux proposer un covoiturage",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                Switch(checked = hasCar, onCheckedChange = { hasCar = it })
            }

            Spacer(Modifier.height(4.dp))

            Text(
                text = "Mon emploi du temps de navette",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            Text(
                text = "Cochez, pour chaque jour, la navette du matin et/ou du soir.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            Weekday.entries.forEach { day ->
                ScheduleDayRow(
                    day = day,
                    isMorning = selected[day to Period.MORNING] ?: false,
                    isEvening = selected[day to Period.EVENING] ?: false,
                    onToggleMorning = { toggle(day, Period.MORNING) },
                    onToggleEvening = { toggle(day, Period.EVENING) }
                )
            }

            Spacer(Modifier.height(24.dp))
        }
    }
}

/** Menu déroulant pour choisir la matière enseignée. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun SubjectDropdown(
    selected: String,
    onSelected: (String) -> Unit
) {
    var expanded by remember { mutableStateOf(false) }
    ExposedDropdownMenuBox(
        expanded = expanded,
        onExpandedChange = { expanded = it },
        modifier = Modifier.fillMaxWidth()
    ) {
        OutlinedTextField(
            value = selected,
            onValueChange = {},
            readOnly = true,
            label = { Text("Matière") },
            placeholder = { Text("Choisir une matière") },
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
            modifier = Modifier
                .menuAnchor()
                .fillMaxWidth()
        )
        ExposedDropdownMenu(
            expanded = expanded,
            onDismissRequest = { expanded = false }
        ) {
            Subjects.ALL.forEach { subject ->
                DropdownMenuItem(
                    text = { Text(subject) },
                    onClick = {
                        onSelected(subject)
                        expanded = false
                    }
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ScheduleDayRow(
    day: Weekday,
    isMorning: Boolean,
    isEvening: Boolean,
    onToggleMorning: () -> Unit,
    onToggleEvening: () -> Unit
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Text(
            text = day.label,
            modifier = Modifier.width(96.dp),
            style = MaterialTheme.typography.bodyLarge
        )
        FilterChip(
            selected = isMorning,
            onClick = onToggleMorning,
            label = { Text(Period.MORNING.label) },
            leadingIcon = if (isMorning) {
                { Icon(Icons.Filled.Check, contentDescription = null, modifier = Modifier.height(18.dp)) }
            } else null
        )
        FilterChip(
            selected = isEvening,
            onClick = onToggleEvening,
            label = { Text(Period.EVENING.label) },
            leadingIcon = if (isEvening) {
                { Icon(Icons.Filled.Check, contentDescription = null, modifier = Modifier.height(18.dp)) }
            } else null
        )
    }
}
