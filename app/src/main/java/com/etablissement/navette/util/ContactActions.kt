package com.etablissement.navette.util

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.widget.Toast

/**
 * Actions de communication avec un enseignant via son numéro de téléphone.
 * Utilise les intents Android (aucun serveur de messagerie requis).
 */
object ContactActions {

    /** Ouvre l'application téléphone avec le numéro pré-rempli. */
    fun call(context: Context, phone: String) {
        if (phone.isBlank()) {
            toastNoPhone(context); return
        }
        val intent = Intent(Intent.ACTION_DIAL, Uri.parse("tel:${sanitize(phone)}"))
        launch(context, intent)
    }

    /** Ouvre l'application SMS avec le numéro pré-rempli. */
    fun sms(context: Context, phone: String) {
        if (phone.isBlank()) {
            toastNoPhone(context); return
        }
        val intent = Intent(Intent.ACTION_SENDTO, Uri.parse("smsto:${sanitize(phone)}"))
        launch(context, intent)
    }

    /**
     * Ouvre WhatsApp (via wa.me) avec le numéro. Le numéro doit idéalement être
     * au format international (ex: 2126XXXXXXXX). On retire les caractères non
     * numériques et le '+' éventuel.
     */
    fun whatsapp(context: Context, phone: String) {
        if (phone.isBlank()) {
            toastNoPhone(context); return
        }
        val number = sanitize(phone).removePrefix("+").filter { it.isDigit() }
        val intent = Intent(Intent.ACTION_VIEW, Uri.parse("https://wa.me/$number"))
        launch(context, intent, errorMessage = "WhatsApp n'est pas disponible")
    }

    private fun sanitize(phone: String): String =
        phone.filter { it.isDigit() || it == '+' }

    private fun launch(
        context: Context,
        intent: Intent,
        errorMessage: String = "Aucune application disponible pour cette action"
    ) {
        try {
            context.startActivity(intent)
        } catch (e: Exception) {
            Toast.makeText(context, errorMessage, Toast.LENGTH_SHORT).show()
        }
    }

    private fun toastNoPhone(context: Context) {
        Toast.makeText(context, "Numéro de téléphone non renseigné", Toast.LENGTH_SHORT).show()
    }
}
