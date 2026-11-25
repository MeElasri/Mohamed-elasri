package com.elasri.presence.util

import java.time.LocalDate
import java.time.LocalTime
import java.time.format.DateTimeFormatter

/**
 * Builds what is sent to the school administration:
 * a short message (readable directly in WhatsApp/email) + an Excel attachment.
 */
object AdminReport {

    data class Line(
        val className: String,
        val period: String,
        val name: String,
        val massarCode: String,
        val status: String // "Absent" or "Retard"
    )

    /** One roll call (or several) for a given day. */
    data class CallInfo(val className: String, val period: String, val present: Int, val total: Int)

    data class Output(
        val subject: String,
        val message: String,
        val fileName: String,
        val xlsx: ByteArray,
        /** Printable A4 document (Android print dialog → printer or PDF). */
        val html: String
    )

    private fun safeFile(s: String) = s.replace(Regex("[^A-Za-z0-9_-]+"), "_").trim('_').ifBlank { "classe" }

    // ---------------- Printable HTML ----------------

    internal fun esc(s: String): String = s
        .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\"", "&quot;")

    private fun statusCell(status: String): String {
        val cls = when (status) { "Absent" -> "A"; "Retard" -> "R"; else -> "" }
        return """<td class="$cls">${esc(status)}</td>"""
    }

    private fun codeHtml(l: Line) =
        if (l.massarCode.isBlank()) "" else """ <span class="code">(${esc(l.massarCode)})</span>"""

    /** Wraps a report body in a clean, print-ready A4 page. */
    internal fun page(title: String, subtitle: String, body: String): String = """
<!doctype html><html lang="fr"><head><meta charset="utf-8">
<style>
 @page { margin: 14mm; }
 *{box-sizing:border-box;}
 body{font-family:-apple-system,Roboto,"Helvetica Neue",Arial,sans-serif;color:#1b2430;font-size:12px;margin:0;}
 h1{font-size:18px;color:#13315C;margin:0 0 2px;}
 .sub{color:#5F6B7A;font-size:12px;margin:0 0 12px;}
 h2{font-size:13px;color:#16888A;margin:16px 0 4px;}
 p.tot{font-size:13px;margin:4px 0 2px;}
 table{width:100%;border-collapse:collapse;margin:4px 0 10px;}
 th,td{border:1px solid #d7dde5;padding:4px 7px;text-align:left;font-size:11px;vertical-align:top;}
 th{background:#eef2f6;color:#13315C;}
 td.num{text-align:center;color:#13315C;}
 .code{color:#8a93a0;}
 .A{color:#BA1A1A;font-weight:bold;}
 .R{color:#EF6C00;font-weight:bold;}
 .foot{margin-top:16px;color:#8a93a0;font-size:10px;border-top:1px solid #e2e6ec;padding-top:6px;}
</style></head><body>
<h1>${esc(title)}</h1>
<p class="sub">${esc(subtitle)}</p>
$body
<p class="foot">Présence Classe — document imprimé le ${LocalDate.now().fmtShort()}</p>
</body></html>
""".trimIndent()

    private fun teacherLine(teacher: String) = if (teacher.isBlank()) "" else "Enseignant : $teacher\n"

    /** Report for one roll call. */
    fun forSession(
        teacher: String,
        date: LocalDate,
        call: CallInfo,
        lines: List<Line>,
        sentAt: LocalTime = LocalTime.now()
    ): Output {
        val absents = lines.filter { it.status == "Absent" }
        val lates = lines.filter { it.status == "Retard" }
        val slot = if (call.period.isBlank()) "" else " · ${call.period}"

        val msg = buildString {
            append("📋 Absences – ${call.className}\n")
            append("${date.fmtLong()}$slot\n")
            append(teacherLine(teacher))
            append("\n")
            append("❌ Absents (${absents.size})")
            if (absents.isEmpty()) append(" : aucun\n") else {
                append(" :\n"); absents.forEachIndexed { i, l -> append("${i + 1}. ${l.name}${code(l)}\n") }
            }
            append("⏰ Retards (${lates.size})")
            if (lates.isEmpty()) append(" : aucun\n") else {
                append(" :\n"); lates.forEachIndexed { i, l -> append("${i + 1}. ${l.name}${code(l)}\n") }
            }
            append("✅ Présents : ${call.present}/${call.total}")
        }

        val rows = mutableListOf(
            listOf("Absences – ${call.className}"),
            listOf("Date", date.fmtShort()),
            listOf("Séance", call.period.ifBlank { "-" }),
            listOf("Enseignant", teacher.ifBlank { "-" }),
            listOf("Présents", "${call.present}/${call.total}"),
            listOf("Envoyé à", sentAt.format(DateTimeFormatter.ofPattern("HH:mm"))),
            emptyList(),
            listOf("N°", "Nom", "Code Massar", "Statut")
        )
        lines.forEachIndexed { i, l -> rows += listOf("${i + 1}", l.name, l.massarCode, l.status) }
        if (lines.isEmpty()) rows += listOf("", "Aucune absence ni retard")

        val xlsx = XlsxWriter.write(listOf(XlsxWriter.Sheet(call.className, rows, boldRows = setOf(0, 7))))
        val slotFile = if (call.period.isBlank()) "" else "_" + safeFile(call.period)

        val sub = listOfNotNull(
            date.fmtLong() + slot,
            teacher.takeIf { it.isNotBlank() }?.let { "Enseignant : $it" },
            "Présents : ${call.present}/${call.total}"
        ).joinToString(" · ")
        val bodyRows = if (lines.isEmpty()) """<tr><td colspan="4">Aucune absence ni retard.</td></tr>"""
        else buildString {
            lines.forEachIndexed { i, l ->
                append("""<tr><td class="num">${i + 1}</td><td>${esc(l.name)}</td><td>${esc(l.massarCode)}</td>${statusCell(l.status)}</tr>""")
            }
        }
        val html = page(
            "Absences – ${call.className}", sub,
            """<table><tr><th>N°</th><th>Nom</th><th>Code Massar</th><th>Statut</th></tr>$bodyRows</table>"""
        )

        return Output(
            subject = "Absences ${call.className} – ${date.fmtShort()}${slot}",
            message = msg,
            fileName = "absences_${safeFile(call.className)}_${date}$slotFile.xlsx",
            xlsx = xlsx,
            html = html
        )
    }

    /** All classes for one day (one sheet "Toutes les classes" + one sheet per class). */
    fun forDay(teacher: String, date: LocalDate, calls: List<CallInfo>, lines: List<Line>): Output {
        val msg = buildString {
            append("📋 Absences du ${date.fmtLong()}\n")
            append(teacherLine(teacher))
            if (calls.isEmpty()) {
                append("\nAucun appel enregistré ce jour.")
                return@buildString
            }
            for (c in calls.sortedWith(compareBy({ it.period }, { it.className }))) {
                val cl = lines.filter { it.className == c.className && it.period == c.period }
                val abs = cl.filter { it.status == "Absent" }
                val late = cl.filter { it.status == "Retard" }
                val slot = if (c.period.isBlank()) "" else " (${c.period})"
                append("\n▶ ${c.className}$slot – ${abs.size} absent(s), ${late.size} retard(s)\n")
                abs.forEach { append("   ❌ ${it.name}${code(it)}\n") }
                late.forEach { append("   ⏰ ${it.name}${code(it)}\n") }
            }
        }.trimEnd()

        val header = listOf("Classe", "Séance", "Nom", "Code Massar", "Statut")
        val all = mutableListOf(
            listOf("Absences du ${date.fmtShort()}" + if (teacher.isBlank()) "" else " – $teacher"),
            emptyList(),
            header
        )
        lines.sortedWith(compareBy({ it.className }, { it.period }, { it.name }))
            .forEach { all += listOf(it.className, it.period, it.name, it.massarCode, it.status) }
        if (lines.isEmpty()) all += listOf("Aucune absence ni retard")

        val summary = mutableListOf(listOf("Classe", "Séance", "Présents", "Absents", "Retards"))
        calls.sortedWith(compareBy({ it.className }, { it.period })).forEach { c ->
            val cl = lines.filter { it.className == c.className && it.period == c.period }
            summary += listOf(
                c.className, c.period, "${c.present}/${c.total}",
                cl.count { it.status == "Absent" }.toString(), cl.count { it.status == "Retard" }.toString()
            )
        }

        val xlsx = XlsxWriter.write(
            listOf(
                XlsxWriter.Sheet("Absences", all, boldRows = setOf(0, 2)),
                XlsxWriter.Sheet("Résumé", summary)
            )
        )

        val html = run {
            val sub = date.fmtLong() + if (teacher.isBlank()) "" else " · $teacher"
            val body = buildString {
                if (calls.isEmpty()) {
                    append("<p>Aucun appel enregistré ce jour.</p>")
                } else {
                    append("<h2>Résumé par séance</h2>")
                    append("<table><tr><th>Classe</th><th>Séance</th><th>Présents</th><th>Absents</th><th>Retards</th></tr>")
                    calls.sortedWith(compareBy({ it.className }, { it.period })).forEach { c ->
                        val cl = lines.filter { it.className == c.className && it.period == c.period }
                        append(
                            "<tr><td>${esc(c.className)}</td><td>${esc(c.period)}</td>" +
                                """<td class="num">${c.present}/${c.total}</td>""" +
                                """<td class="num">${cl.count { it.status == "Absent" }}</td>""" +
                                """<td class="num">${cl.count { it.status == "Retard" }}</td></tr>"""
                        )
                    }
                    append("</table>")
                    append("<h2>Détail des absences et retards</h2>")
                    append("<table><tr><th>Classe</th><th>Séance</th><th>Nom</th><th>Statut</th></tr>")
                    if (lines.isEmpty()) {
                        append("""<tr><td colspan="4">Aucune absence ni retard.</td></tr>""")
                    } else {
                        lines.sortedWith(compareBy({ it.className }, { it.period }, { it.name })).forEach { l ->
                            append(
                                "<tr><td>${esc(l.className)}</td><td>${esc(l.period)}</td>" +
                                    "<td>${esc(l.name)}${codeHtml(l)}</td>${statusCell(l.status)}</tr>"
                            )
                        }
                    }
                    append("</table>")
                }
            }
            page("Absences du ${date.fmtShort()}", sub, body)
        }

        return Output(
            subject = "Absences du ${date.fmtShort()}" + if (teacher.isBlank()) "" else " – $teacher",
            message = msg,
            fileName = "absences_${date}${if (teacher.isBlank()) "" else "_" + safeFile(teacher)}.xlsx",
            xlsx = xlsx,
            html = html
        )
    }

    private fun code(l: Line) = if (l.massarCode.isBlank()) "" else " (${l.massarCode})"
}
