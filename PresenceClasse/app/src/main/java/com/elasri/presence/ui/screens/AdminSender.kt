package com.elasri.presence.ui.screens

import android.widget.Toast
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.platform.LocalContext
import com.elasri.presence.data.frMessage
import com.elasri.presence.util.AdminReport
import com.elasri.presence.util.printHtml
import com.elasri.presence.util.shareAdminReport
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.launch

/** Sends a report (message + Excel file) through the share sheet (WhatsApp, e-mail…). */
class AdminSender internal constructor(private val start: (suspend () -> AdminReport.Output, () -> Unit) -> Unit) {
    /** [after] runs once the share sheet has been opened (or on error). */
    fun send(after: () -> Unit = {}, build: suspend () -> AdminReport.Output) = start(build, after)
}

@Composable
fun rememberAdminSender(): AdminSender {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    return remember {
        AdminSender { build, after ->
            scope.launch {
                try {
                    context.shareAdminReport(build())
                } catch (e: CancellationException) {
                    throw e // screen closed: don't show an error, don't navigate again
                } catch (e: Exception) {
                    Toast.makeText(context, "Envoi impossible : ${e.frMessage()}", Toast.LENGTH_LONG).show()
                }
                after()
            }
        }
    }
}

/** Opens the Android print dialog (printer or "Enregistrer en PDF") for a report. */
class AdminPrinter internal constructor(private val start: (suspend () -> AdminReport.Output) -> Unit) {
    fun print(build: suspend () -> AdminReport.Output) = start(build)
}

@Composable
fun rememberAdminPrinter(): AdminPrinter {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    return remember {
        AdminPrinter { build ->
            scope.launch {
                try {
                    val out = build()
                    context.printHtml(out.subject, out.html)
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Exception) {
                    Toast.makeText(context, "Impression impossible : ${e.frMessage()}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }
}
