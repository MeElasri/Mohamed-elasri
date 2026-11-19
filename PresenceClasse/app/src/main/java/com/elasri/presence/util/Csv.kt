package com.elasri.presence.util

import java.nio.charset.Charset

/**
 * Minimal CSV reader/writer tuned for files exported from Excel:
 * - separator auto-detected (; , or tab), UTF-8 with or without BOM, Windows-1252 fallback
 * - student detection is shared with the .xlsx import (see [StudentTable])
 */
object Csv {

    fun decode(bytes: ByteArray): String {
        val utf8 = String(bytes, Charsets.UTF_8)
        val text = if (utf8.contains('�')) String(bytes, Charset.forName("windows-1252")) else utf8
        return text.removePrefix("﻿")
    }

    private fun detectSeparator(lines: List<String>): Char =
        listOf(';', ',', '\t').maxBy { sep -> lines.take(20).sumOf { line -> line.count { it == sep } } }

    fun splitLine(line: String, sep: Char): List<String> {
        val out = mutableListOf<String>()
        val cur = StringBuilder()
        var inQuotes = false
        var i = 0
        while (i < line.length) {
            val c = line[i]
            when {
                inQuotes && c == '"' && i + 1 < line.length && line[i + 1] == '"' -> {
                    cur.append('"'); i++
                }
                c == '"' -> inQuotes = !inQuotes
                c == sep && !inQuotes -> {
                    out += cur.toString(); cur.clear()
                }
                else -> cur.append(c)
            }
            i++
        }
        out += cur.toString()
        return out.map { it.trim() }
    }

    fun rows(text: String): List<List<String>> {
        val lines = text.lines().filter { it.isNotBlank() }
        if (lines.isEmpty()) return emptyList()
        val sep = detectSeparator(lines)
        return lines.map { splitLine(it, sep) }
    }

    fun parseStudents(text: String): List<StudentTable.ParsedStudent> =
        StudentTable.extractStudents(rows(text))

    private fun escape(v: String, sep: Char): String =
        if (v.any { it == sep || it == '"' || it == '\n' || it == '\r' })
            "\"" + v.replace("\"", "\"\"") + "\""
        else v

    /** UTF-8 with BOM and ';' so Excel (French locale) opens it correctly. */
    fun build(header: List<String>, rows: List<List<String>>, sep: Char = ';'): ByteArray {
        val sb = StringBuilder("﻿")
        (listOf(header) + rows).forEach { row ->
            sb.append(row.joinToString(sep.toString()) { escape(it, sep) }).append("\r\n")
        }
        return sb.toString().toByteArray(Charsets.UTF_8)
    }
}
