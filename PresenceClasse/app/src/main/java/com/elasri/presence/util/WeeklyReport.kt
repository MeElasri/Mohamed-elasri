package com.elasri.presence.util

import com.elasri.presence.data.WeeklyStats
import java.util.Locale

/** Weekly statistics → message + Excel file (3 sheets) for the administration. */
object WeeklyReport {

    private fun hours(h: Double?): String =
        if (h == null) "" else String.format(Locale.FRENCH, "%.1f", h).removeSuffix(",0") + " h"

    fun build(r: WeeklyStats.Result, threshold: Int, maxInMessage: Int = 40): AdminReport.Output {
        val convoke = r.toConvoke(threshold)
        val period = "du ${r.from.fmtShort()} au ${r.to.fmtShort()}"

        val msg = buildString {
            append("📊 Absences de la semaine $period\n")
            append("${r.sessions} appel(s) · ${r.absences} absence(s) · ${r.lates} retard(s)\n\n")
            append("👪 Parents à convoquer (≥ $threshold absence(s)) : ${convoke.size}\n")
            convoke.take(maxInMessage).forEachIndexed { i, s ->
                val h = hours(s.absentHours).let { if (it.isBlank()) "" else ", $it" }
                append("${i + 1}. ${s.name} – ${s.className} : ${s.absences} abs.$h\n")
            }
            if (convoke.size > maxInMessage) append("… et ${convoke.size - maxInMessage} autre(s) (voir le fichier Excel)\n")
            if (r.classes.isNotEmpty()) {
                append("\n🏫 Par classe :\n")
                r.classes.forEach { append("• ${it.className} : ${it.absences} abs., ${it.lates} ret. (${it.sessions} appels)\n") }
            }
        }.trimEnd()

        val convokeRows = mutableListOf(
            listOf("Parents à convoquer – semaine $period"),
            listOf("Seuil : $threshold absence(s) ou plus dans la semaine"),
            emptyList(),
            listOf("N°", "Classe", "Nom", "Code Massar", "Absences", "Heures d'absence", "Retards", "Détail des absences")
        )
        convoke.forEachIndexed { i, s ->
            convokeRows += listOf(
                "${i + 1}", s.className, s.name, s.massarCode, "${s.absences}", hours(s.absentHours), "${s.lates}", s.absenceDetails()
            )
        }
        if (convoke.isEmpty()) convokeRows += listOf("", "Aucun élève n'atteint le seuil cette semaine.")

        val studentRows = mutableListOf(
            listOf("Classe", "Nom", "Code Massar", "Absences", "Heures d'absence", "Retards", "Détail des absences", "À convoquer")
        )
        r.students.forEach { s ->
            studentRows += listOf(
                s.className, s.name, s.massarCode, "${s.absences}", hours(s.absentHours), "${s.lates}",
                s.absenceDetails(), if (s.absences >= threshold) "OUI" else ""
            )
        }

        val classRows = mutableListOf(listOf("Classe", "Appels", "Absences", "Retards", "Élèves absents au moins une fois"))
        r.classes.forEach { c ->
            classRows += listOf(c.className, "${c.sessions}", "${c.absences}", "${c.lates}", "${c.studentsAbsent}")
        }
        classRows += listOf("TOTAL", "${r.sessions}", "${r.absences}", "${r.lates}", "${r.students.count { it.absences > 0 }}")

        val xlsx = XlsxWriter.write(
            listOf(
                XlsxWriter.Sheet("Parents à convoquer", convokeRows, boldRows = setOf(0, 3)),
                XlsxWriter.Sheet("Tous les élèves", studentRows),
                XlsxWriter.Sheet("Par classe", classRows, boldRows = setOf(0, classRows.size - 1))
            )
        )
        return AdminReport.Output(
            subject = "Absences de la semaine $period – ${convoke.size} parent(s) à convoquer",
            message = msg,
            fileName = "absences_semaine_${r.from}.xlsx",
            xlsx = xlsx
        )
    }
}
