package com.elasri.presence.util

import java.io.ByteArrayOutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * Writes a small, valid .xlsx (no library). Text cells only, bold first row of each table,
 * auto column widths. Arabic and French names are kept intact (UTF-8 XML),
 * which CSV does not guarantee in every Excel version.
 */
object XlsxWriter {

    /** [boldRows] = 0-based row indexes rendered in bold (titles, headers). */
    data class Sheet(val name: String, val rows: List<List<String>>, val boldRows: Set<Int> = setOf(0))

    fun write(sheets: List<Sheet>): ByteArray {
        val safe = uniqueNames(sheets)
        val out = ByteArrayOutputStream()
        ZipOutputStream(out).use { zip ->
            fun put(path: String, xml: String) {
                zip.putNextEntry(ZipEntry(path))
                zip.write(xml.toByteArray(Charsets.UTF_8))
                zip.closeEntry()
            }
            put("[Content_Types].xml", contentTypes(safe.size))
            put("_rels/.rels", ROOT_RELS)
            put("xl/workbook.xml", workbook(safe.map { it.name }))
            put("xl/_rels/workbook.xml.rels", workbookRels(safe.size))
            put("xl/styles.xml", STYLES)
            safe.forEachIndexed { i, s -> put("xl/worksheets/sheet${i + 1}.xml", sheetXml(s)) }
        }
        return out.toByteArray()
    }

    // ---------------- XML parts ----------------

    private const val HEAD = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>"""
    private const val NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    private const val NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

    private val ROOT_RELS = HEAD +
        """<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">""" +
        """<Relationship Id="rId1" Type="$NS_REL/officeDocument" Target="xl/workbook.xml"/>""" +
        "</Relationships>"

    private val STYLES = HEAD + """<styleSheet xmlns="$NS_MAIN">""" +
        """<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>""" +
        """<font><b/><sz val="11"/><name val="Calibri"/></font></fonts>""" +
        """<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>""" +
        """<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>""" +
        """<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>""" +
        """<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>""" +
        """<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs>""" +
        """<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>""" +
        "</styleSheet>"

    private fun contentTypes(n: Int) = HEAD +
        """<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">""" +
        """<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>""" +
        """<Default Extension="xml" ContentType="application/xml"/>""" +
        """<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>""" +
        """<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>""" +
        (1..n).joinToString("") {
            """<Override PartName="/xl/worksheets/sheet$it.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>"""
        } + "</Types>"

    private fun workbook(names: List<String>) = HEAD +
        """<workbook xmlns="$NS_MAIN" xmlns:r="$NS_REL"><sheets>""" +
        names.mapIndexed { i, n -> """<sheet name="${esc(n)}" sheetId="${i + 1}" r:id="rId${i + 1}"/>""" }
            .joinToString("") +
        "</sheets></workbook>"

    private fun workbookRels(n: Int) = HEAD +
        """<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">""" +
        (1..n).joinToString("") {
            """<Relationship Id="rId$it" Type="$NS_REL/worksheet" Target="worksheets/sheet$it.xml"/>"""
        } +
        """<Relationship Id="rId${n + 1}" Type="$NS_REL/styles" Target="styles.xml"/>""" +
        "</Relationships>"

    private fun sheetXml(s: Sheet): String {
        val sb = StringBuilder(HEAD).append("""<worksheet xmlns="$NS_MAIN">""")
        val nCols = s.rows.maxOfOrNull { it.size } ?: 0
        if (nCols > 0) {
            sb.append("<cols>")
            for (c in 0 until nCols) {
                // width from the longest value, ignoring long single-cell title lines
                val w = s.rows.filter { it.size > 1 }.maxOfOrNull { it.getOrNull(c)?.length ?: 0 } ?: 10
                sb.append("""<col min="${c + 1}" max="${c + 1}" width="${(w + 3).coerceIn(8, 60)}" customWidth="1"/>""")
            }
            sb.append("</cols>")
        }
        sb.append("<sheetData>")
        s.rows.forEachIndexed { r, row ->
            sb.append("""<row r="${r + 1}">""")
            row.forEachIndexed { c, v ->
                if (v.isNotEmpty()) {
                    val style = if (r in s.boldRows) """ s="1"""" else ""
                    val ref = "${colName(c)}${r + 1}"
                    if (INTEGER.matches(v)) {
                        // real numbers, so the administration can sort and sum them in Excel
                        sb.append("""<c r="$ref"$style><v>$v</v></c>""")
                    } else {
                        sb.append("""<c r="$ref" t="inlineStr"$style><is><t xml:space="preserve">""")
                            .append(esc(v)).append("</t></is></c>")
                    }
                }
            }
            sb.append("</row>")
        }
        sb.append("</sheetData></worksheet>")
        return sb.toString()
    }

    // ---------------- helpers ----------------

    /** Plain integers without leading zeros (codes such as "0123" stay text). */
    private val INTEGER = Regex("^(0|[1-9][0-9]{0,8})$")

    fun colName(i: Int): String {
        var n = i + 1
        val sb = StringBuilder()
        while (n > 0) {
            val r = (n - 1) % 26
            sb.insert(0, ('A' + r))
            n = (n - 1) / 26
        }
        return sb.toString()
    }

    private fun esc(s: String): String {
        val sb = StringBuilder(s.length)
        for (ch in s) {
            when {
                ch == '&' -> sb.append("&amp;")
                ch == '<' -> sb.append("&lt;")
                ch == '>' -> sb.append("&gt;")
                ch == '"' -> sb.append("&quot;")
                ch == '\t' || ch == '\n' || ch == '\r' -> sb.append(ch)
                ch < ' ' || ch == '￾' || ch == '￿' -> {} // invalid in XML
                else -> sb.append(ch)
            }
        }
        return sb.toString()
    }

    /** Excel sheet names: max 31 chars, no []:*?/\ , unique. */
    private fun uniqueNames(sheets: List<Sheet>): List<Sheet> {
        val used = mutableSetOf<String>()
        return sheets.mapIndexed { i, s ->
            val base = s.name.replace(Regex("""[\[\]:*?/\\]"""), "-").trim().ifBlank { "Feuille${i + 1}" }.take(31)
            var name = base
            var k = 2
            while (name.lowercase() in used) {
                val suffix = " ($k)"
                name = base.take(31 - suffix.length) + suffix
                k++
            }
            used += name.lowercase()
            s.copy(name = name)
        }
    }
}
