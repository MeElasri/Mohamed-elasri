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
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.DirectionsCar
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.PersonOutline
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.etablissement.navette.data.model.Group
import com.etablissement.navette.data.model.GroupPeriod
import com.etablissement.navette.data.model.Teacher
import com.etablissement.navette.ui.NavetteViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun GroupsScreen(
    viewModel: NavetteViewModel,
    onCreateGroup: () -> Unit,
    onEditGroup: (String) -> Unit
) {
    val groups by viewModel.groups.collectAsStateWithLifecycle()
    val teachers by viewModel.teachers.collectAsStateWithLifecycle()
    val myProfile by viewModel.myProfile.collectAsStateWithLifecycle()
    val myUid = viewModel.currentUid
    val canCreate = myProfile?.hasCar == true

    var groupToDelete by remember { mutableStateOf<Group?>(null) }

    Scaffold(
        topBar = { TopAppBar(title = { Text("Groupes de navette") }) },
        floatingActionButton = {
            if (canCreate) {
                FloatingActionButton(onClick = onCreateGroup) {
                    Icon(Icons.Filled.Add, contentDescription = "Créer un groupe")
                }
            }
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp)
        ) {
            if (!canCreate) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.secondaryContainer
                    )
                ) {
                    Text(
                        text = "Seuls les enseignants qui possèdent une voiture peuvent créer un " +
                            "groupe. Activez « Je possède une voiture » dans Mon profil pour en créer un.",
                        modifier = Modifier.padding(12.dp),
                        style = MaterialTheme.typography.bodySmall
                    )
                }
                Spacer(Modifier.height(12.dp))
            }

            if (groups.isEmpty()) {
                EmptyState(
                    message = "Aucun groupe de navette pour le moment.\n" +
                        if (canCreate) "Appuyez sur + pour créer votre groupe."
                        else "Un conducteur pourra bientôt en créer un."
                )
            } else {
                LazyColumn(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    items(groups, key = { it.id }) { group ->
                        GroupCard(
                            group = group,
                            driver = teachers.firstOrNull { it.id == group.driverId },
                            members = viewModel.teachersByIds(group.memberIds),
                            isMine = group.driverId == myUid,
                            onEdit = { onEditGroup(group.id) },
                            onDelete = { groupToDelete = group }
                        )
                    }
                }
            }
        }
    }

    groupToDelete?.let { group ->
        AlertDialog(
            onDismissRequest = { groupToDelete = null },
            title = { Text("Supprimer le groupe ?") },
            text = { Text("Supprimer définitivement « ${group.name} » ?") },
            confirmButton = {
                TextButton(onClick = {
                    viewModel.deleteGroup(group.id)
                    groupToDelete = null
                }) { Text("Supprimer") }
            },
            dismissButton = {
                TextButton(onClick = { groupToDelete = null }) { Text("Annuler") }
            }
        )
    }
}

@Composable
private fun GroupCard(
    group: Group,
    driver: Teacher?,
    members: List<Teacher>,
    isMine: Boolean,
    onEdit: () -> Unit,
    onDelete: () -> Unit
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = group.name,
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.weight(1f)
                )
                if (isMine) {
                    IconButton(onClick = onEdit) {
                        Icon(Icons.Filled.Edit, contentDescription = "Modifier")
                    }
                    IconButton(onClick = onDelete) {
                        Icon(Icons.Filled.Delete, contentDescription = "Supprimer")
                    }
                }
            }
            Text(
                text = "Navette : ${GroupPeriod.fromName(group.period).label}",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            // Places : 5 au total (conducteur compris). Un membre indisponible libère sa place.
            val driverAvailable = driver?.available ?: true
            val unresolved = group.memberIds.size - members.size
            val occupied = 1 + members.count { it.available } + unresolved
            val freeSeats = (Group.CAPACITY - occupied).coerceAtLeast(0)
            val absentCount = members.count { !it.available }

            Spacer(Modifier.height(8.dp))
            val (seatText, seatColor) = when {
                !driverAvailable ->
                    "Conducteur indisponible : pas de navette pour ce groupe" to
                        MaterialTheme.colorScheme.error
                freeSeats > 0 ->
                    ("$freeSeats place${if (freeSeats > 1) "s" else ""} disponible" +
                        (if (freeSeats > 1) "s" else "") +
                        (if (absentCount > 0) " ($absentCount absent${if (absentCount > 1) "s" else ""})" else "") +
                        " — contactez le conducteur") to Color(0xFF2E7D32)
                else -> "Complet ($occupied/${Group.CAPACITY})" to
                    MaterialTheme.colorScheme.onSurfaceVariant
            }
            Text(
                text = seatText,
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.Bold,
                color = seatColor
            )

            Spacer(Modifier.height(12.dp))

            // Conducteur
            Text(
                text = "Conducteur",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.primary
            )
            PersonRow(
                name = driver?.name ?: group.driverName,
                subtitle = driver?.subject.orEmpty(),
                phone = driver?.phone.orEmpty(),
                isDriver = true,
                available = driver?.available ?: true
            )

            if (members.isNotEmpty()) {
                Spacer(Modifier.height(8.dp))
                Text(
                    text = "Membres (${members.size})",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.primary
                )
                members.forEach { member ->
                    PersonRow(
                        name = member.name,
                        subtitle = member.subject,
                        phone = member.phone,
                        isDriver = false,
                        available = member.available
                    )
                }
            }
        }
    }
}

@Composable
private fun PersonRow(
    name: String,
    subtitle: String,
    phone: String,
    isDriver: Boolean,
    available: Boolean = true
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            imageVector = if (isDriver) Icons.Filled.DirectionsCar else Icons.Filled.PersonOutline,
            contentDescription = null
        )
        Spacer(Modifier.width(12.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = name.ifBlank { "(sans nom)" },
                style = MaterialTheme.typography.bodyLarge,
                fontWeight = FontWeight.SemiBold
            )
            if (subtitle.isNotBlank()) {
                Text(
                    text = subtitle,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            if (!available) {
                Text(
                    text = "● Indisponible",
                    style = MaterialTheme.typography.labelSmall,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.error
                )
            }
        }
        if (phone.isNotBlank()) {
            ContactButtons(phone = phone)
        }
    }
}
