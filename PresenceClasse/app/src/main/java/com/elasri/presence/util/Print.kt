package com.elasri.presence.util

import android.content.Context
import android.print.PrintAttributes
import android.print.PrintManager
import android.webkit.WebView
import android.webkit.WebViewClient

// Strong reference kept until the print job has been handed to the system (then released).
private var printingWebView: WebView? = null

/**
 * Opens the Android print dialog for an HTML document. The user can pick a printer
 * or "Enregistrer au format PDF". Works offline, no extra permission needed.
 */
fun Context.printHtml(jobName: String, html: String) {
    val name = jobName.ifBlank { "Rapport" }
    val webView = WebView(this)
    webView.webViewClient = object : WebViewClient() {
        override fun onPageFinished(view: WebView, url: String?) {
            val printManager = getSystemService(Context.PRINT_SERVICE) as PrintManager
            val adapter = view.createPrintDocumentAdapter(name)
            printManager.print(name, adapter, PrintAttributes.Builder().build())
            printingWebView = null // released once the system holds the adapter
        }
    }
    printingWebView = webView
    webView.loadDataWithBaseURL(null, html, "text/html", "UTF-8", null)
}
