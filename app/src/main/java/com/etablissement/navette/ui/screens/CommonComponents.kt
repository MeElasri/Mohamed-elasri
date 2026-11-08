package com.etablissement.navette.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Call
import androidx.compose.material.icons.filled.Sms
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalIconButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.etablissement.navette.R
import com.etablissement.navette.util.ContactActions

/**
 * Rangée horizontale défilable de "chips" sélectionnables.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun <T> LazyRowChips(
    items: List<T>,
    isSelected: (T) -> Boolean,
    label: (T) -> String,
    onClick: (T) -> Unit
) {
    LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        items(items) { item ->
            FilterChip(
                selected = isSelected(item),
                onClick = { onClick(item) },
                label = { Text(label(item)) }
            )
        }
    }
}

/**
 * Message affiché lorsqu'une liste est vide.
 */
@Composable
fun EmptyState(message: String) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(32.dp),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = message,
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center
        )
    }
}

/**
 * Boutons de contact (Appeler / SMS / WhatsApp) pour un numéro de téléphone.
 * Réutilisé dans l'écran Navette et dans les groupes.
 */
@Composable
fun ContactButtons(phone: String) {
    val context = LocalContext.current
    Row(verticalAlignment = Alignment.CenterVertically) {
        FilledTonalIconButton(onClick = { ContactActions.call(context, phone) }) {
            Icon(Icons.Filled.Call, contentDescription = "Appeler")
        }
        Spacer(Modifier.width(4.dp))
        FilledTonalIconButton(onClick = { ContactActions.sms(context, phone) }) {
            Icon(Icons.Filled.Sms, contentDescription = "SMS")
        }
        Spacer(Modifier.width(4.dp))
        FilledTonalIconButton(
            onClick = { ContactActions.whatsapp(context, phone) },
            colors = IconButtonDefaults.filledTonalIconButtonColors(
                containerColor = Color(0xFF25D366),
                contentColor = Color.White
            )
        ) {
            Icon(
                painter = painterResource(R.drawable.ic_whatsapp),
                contentDescription = "WhatsApp",
                modifier = Modifier.size(20.dp)
            )
        }
    }
}
