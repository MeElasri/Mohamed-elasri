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

    data class Output(val subject: String, val message: String, val fileName: String, val xlsx: ByteArray)

    private fun safeFile(s: String) = s.replace(Regex("[^A-Za-z0-9_-]+"), "_").trim('_').ifBlank { "classe" }

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
        return Output(
            subject = "Absences ${call.className} – ${date.fmtShort()}${slot}",
            message = msg,
            fileName = "absences_${safeFile(call.className)}_${date}$slotFile.xlsx",
            xlsx = xlsx
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
        return Output(
            subject = "Absences du ${date.fmtShort()}" + if (teacher.isBlank()) "" else " – $teacher",
            message = msg,
            fileName = "absences_${date}${if (teacher.isBlank()) "" else "_" + safeFile(teacher)}.xlsx",
            xlsx = xlsx
        )
    }

    private fun code(l: Line) = if (l.massarCode.isBlank()) "" else " (${l.massarCode})"
}
