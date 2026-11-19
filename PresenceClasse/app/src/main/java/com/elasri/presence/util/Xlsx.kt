package com.elasri.presence.util

import org.w3c.dom.Element
import org.w3c.dom.Node
import java.io.ByteArrayInputStream
import java.util.zip.ZipInputStream
import javax.xml.parsers.DocumentBuilderFactory

/**
 * Small .xlsx reader (no external library): returns every sheet as rows of text cells.
 * Supports shared strings, inline strings, numbers and booleans. Formulas give their cached value.
 * Old .xls files (Excel 97-2003) are not supported: re-save them as .xlsx.
 */
object Xlsx {

    data class Sheet(val name: String, val rows: List<List<String>>)

    class NotXlsxException : Exception(
        "Ce fichier n'est pas un classeur .xlsx. Ouvrez-le dans Excel et enregistrez-le au format .xlsx."
    )

    fun isXlsx(bytes: ByteArray): Boolean =
        bytes.size > 4 && bytes[0] == 'P'.code.toByte() && bytes[1] == 'K'.code.toByte()

    fun read(bytes: ByteArray): List<Sheet> {
        if (!isXlsx(bytes)) throw NotXlsxException()
        val files = HashMap<String, ByteArray>()
        ZipInputStream(ByteArrayInputStream(bytes)).use { zip ->
            while (true) {
                val e = zip.nextEntry ?: break
                val n = e.name.removePrefix("/")
                if (n.endsWith(".xml") || n.endsWith(".rels")) files[n] = zip.readBytes()
            }
        }
        val shared = files["xl/sharedStrings.xml"]?.let(::sharedStrings) ?: emptyList()

        // Sheet names + order from workbook.xml, file paths from workbook.xml.rels
        val rels = files["xl/_rels/workbook.xml.rels"]?.let { doc(it) }
            ?.let { d -> elements(d.documentElement, "Relationship") }
            ?.associate { it.getAttribute("Id") to it.getAttribute("Target") } ?: emptyMap()
        val wb = files["xl/workbook.xml"]?.let { doc(it) } ?: throw NotXlsxException()

        return elements(wb.documentElement, "sheet").mapNotNull { s ->
            val rid = s.getAttribute("r:id").ifBlank {
                (0 until s.attributes.length).map { s.attributes.item(it) }
                    .firstOrNull { it.nodeName.endsWith(":id") }?.nodeValue.orEmpty()
            }
            val target = rels[rid] ?: return@mapNotNull null
            val path = when {
                target.startsWith("/") -> target.removePrefix("/")
                target.startsWith("xl/") -> target
                else -> "xl/$target"
            }
            val data = files[path] ?: return@mapNotNull null
            Sheet(s.getAttribute("name"), sheetRows(data, shared))
        }
    }

    // ---------------- internals ----------------

    private fun doc(bytes: ByteArray) = DocumentBuilderFactory.newInstance().apply {
        isNamespaceAware = false
        try {
            setFeature("http://apache.org/xml/features/disallow-doctype-decl", true)
        } catch (ignored: Exception) { /* not supported on every parser */ }
    }.newDocumentBuilder().parse(ByteArrayInputStream(bytes))

    /** All descendant elements whose local name is [name] (prefix-insensitive). */
    private fun elements(root: Element, name: String): List<Element> {
        val out = ArrayList<Element>()
        fun walk(n: Node) {
            var c = n.firstChild
            while (c != null) {
                if (c is Element) {
                    if (c.nodeName == name || c.nodeName.endsWith(":$name")) out += c
                    walk(c)
                }
                c = c.nextSibling
            }
        }
        walk(root)
        return out
    }

    private fun textOf(e: Element, tag: String): String =
        elements(e, tag).joinToString("") { it.textContent ?: "" }

    private fun sharedStrings(bytes: ByteArray): List<String> =
        elements(doc(bytes).documentElement, "si").map { si ->
            // concatenate <t> of all runs, ignore phonetic <rPh> blocks
            val out = StringBuilder()
            elements(si, "t").forEach { t ->
                if (t.parentNode?.nodeName?.endsWith("rPh") != true) out.append(t.textContent)
            }
            out.toString()
        }

    /** "BC12" -> 54 (0-based column index). */
    private fun colIndex(ref: String): Int {
        var n = 0
        for (ch in ref) {
            if (!ch.isLetter()) break
            n = n * 26 + (ch.uppercaseChar() - 'A' + 1)
        }
        return n - 1
    }

    private fun sheetRows(bytes: ByteArray, shared: List<String>): List<List<String>> {
        val rows = ArrayList<List<String>>()
        for (row in elements(doc(bytes).documentElement, "row")) {
            val cells = ArrayList<String>()
            var next = 0
            for (c in elements(row, "c")) {
                val ref = c.getAttribute("r")
                val idx = if (ref.isNotBlank()) colIndex(ref) else next
                val v = textOf(c, "v")
                val value = when (c.getAttribute("t")) {
                    "s" -> v.trim().toIntOrNull()?.let { shared.getOrNull(it) }.orEmpty()
                    "inlineStr" -> textOf(c, "t")
                    "b" -> if (v == "1") "VRAI" else "FAUX"
                    else -> cleanNumber(v)
                }
                while (cells.size < idx) cells += ""
                if (idx < cells.size) cells[idx] = value else cells += value
                next = idx + 1
            }
            rows += cells
        }
        return rows
    }

    /** "12.0" -> "12", keep everything else as-is. */
    private fun cleanNumber(v: String): String =
        if (v.endsWith(".0") && v.dropLast(2).all { it.isDigit() || it == '-' }) v.dropLast(2) else v
}
