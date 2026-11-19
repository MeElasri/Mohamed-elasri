package com.elasri.presence.util

import android.content.ClipData
import android.content.Context
import android.content.Intent
import androidx.core.content.FileProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

private const val XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

/**
 * Opens the Android share sheet (WhatsApp, Gmail, Outlook…) with the message text
 * and the Excel file attached.
 */
suspend fun Context.shareAdminReport(report: AdminReport.Output) {
    val file = withContext(Dispatchers.IO) {
        val dir = File(cacheDir, "shared").apply { mkdirs() }
        dir.listFiles()?.forEach { it.delete() } // keep only the latest report
        File(dir, report.fileName).apply { writeBytes(report.xlsx) }
    }
    val uri = FileProvider.getUriForFile(this, "$packageName.fileprovider", file)
    val send = Intent(Intent.ACTION_SEND).apply {
        type = XLSX_MIME
        putExtra(Intent.EXTRA_STREAM, uri)
        putExtra(Intent.EXTRA_SUBJECT, report.subject)
        putExtra(Intent.EXTRA_TEXT, report.message)
        clipData = ClipData.newRawUri(report.fileName, uri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    val chooser = Intent.createChooser(send, "Envoyer à l'administration").apply {
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    startActivity(chooser)
}
