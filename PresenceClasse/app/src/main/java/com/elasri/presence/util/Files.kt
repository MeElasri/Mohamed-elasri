package com.elasri.presence.util

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/** MIME types offered in the file picker for student lists. */
val STUDENT_LIST_MIME_TYPES = arrayOf(
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", // .xlsx
    "application/vnd.ms-excel",                                          // some apps label .xlsx/.csv this way
    "application/octet-stream",                                          // files from WhatsApp/Drive
    "text/csv", "text/comma-separated-values", "text/plain", "text/*"
)

private const val MAX_FILE_BYTES = 20L * 1024 * 1024

/** Reads a picked file: (display name, bytes). */
suspend fun Context.readPickedFile(uri: Uri): Pair<String, ByteArray> = withContext(Dispatchers.IO) {
    var name = uri.lastPathSegment?.substringAfterLast('/') ?: "liste"
    var size = -1L
    contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE), null, null, null)
        ?.use { c ->
            if (c.moveToFirst()) {
                c.getColumnIndex(OpenableColumns.DISPLAY_NAME).takeIf { it >= 0 }
                    ?.let { i -> c.getString(i)?.let { name = it } }
                c.getColumnIndex(OpenableColumns.SIZE).takeIf { it >= 0 && !c.isNull(it) }
                    ?.let { i -> size = c.getLong(i) }
            }
        }
    require(size <= MAX_FILE_BYTES) { "fichier trop volumineux" }
    val bytes = contentResolver.openInputStream(uri)?.use { it.readBytes() }
        ?: error("fichier inaccessible")
    name to bytes
}
