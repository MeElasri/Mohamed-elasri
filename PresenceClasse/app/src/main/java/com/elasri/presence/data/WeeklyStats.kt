package com.elasri.presence.data

import java.time.DayOfWeek
import java.time.LocalDate
import java.time.format.TextStyle
import java.util.Locale

/**
 * Weekly statistics for the administration (pure logic, unit-testable).
 * Counts are per roll call ("séance"); hours are added when the slot can be read (e.g. "8h-10h").
 */
object WeeklyStats {

    data class Occurrence(val epochDay: Long, val period: String, val teacherName: String, val status: Status)

    data class StudentWeek(
        val classId: String,
        val className: String,
        val studentId: String,
        val name: String,
        val massarCode: String,
        val absences: Int,
        val lates: Int,
        /** Sum of the durations of the absent slots that could be read; null if none could. */
        val absentHours: Double?,
        val occurrences: List<Occurrence>
    ) {
        /** "Lun 8h-10h, Mar 10h-12h" — absences only. */
        fun absenceDetails(): String = occurrences.filter { it.status == Status.ABSENT }
            .sortedBy { it.epochDay }
            .joinToString(", ") { dayShort(it.epochDay) + if (it.period.isBlank()) "" else " ${it.period}" }
    }

    data class ClassWeek(
        val classId: String,
        val className: String,
        val sessions: Int,
        val absences: Int,
        val lates: Int,
        val studentsAbsent: Int
    )

    data class Result(
        val from: LocalDate,
        val to: LocalDate,
        val sessions: Int,
        val absences: Int,
        val lates: Int,
        val classes: List<ClassWeek>,
        /** Every student absent or late at least once, most absences first. */
        val students: List<StudentWeek>
    ) {
        fun toConvoke(threshold: Int): List<StudentWeek> = students.filter { it.absences >= threshold }
    }

    /** Monday of the week containing [d]. */
    fun weekStart(d: LocalDate): LocalDate = d.with(DayOfWeek.MONDAY)

    fun dayShort(epochDay: Long): String =
        LocalDate.ofEpochDay(epochDay).dayOfWeek.getDisplayName(TextStyle.SHORT, Locale.FRENCH)
            .replace(".", "").replaceFirstChar { it.titlecase(Locale.FRENCH) }

    /**
     * Duration in hours of a slot written like "8h-10h", "8h30-10h", "08:00-10:00", "14h - 16h".
     * Returns null when it cannot be read.
     */
    fun periodHours(period: String): Double? {
        val m = Regex("""^\s*(\d{1,2})\s*(?:[h:H]\s*(\d{2})?)?\s*[-–à]+\s*(\d{1,2})\s*(?:[h:H]\s*(\d{2})?)?\s*$""")
            .find(period) ?: return null
        val (h1, m1, h2, m2) = m.destructured
        val start = h1.toInt() * 60 + (m1.toIntOrNull() ?: 0)
        val end = h2.toInt() * 60 + (m2.toIntOrNull() ?: 0)
        val minutes = end - start
        return if (minutes in 1..(8 * 60)) minutes / 60.0 else null
    }

    fun compute(from: LocalDate, to: LocalDate, sessions: List<Session>): Result {
        val inRange = sessions.filter { it.epochDay in from.toEpochDay()..to.toEpochDay() }

        data class Key(val classId: String, val studentId: String)
        class Acc(val className: String, val name: String, val code: String) {
            var abs = 0
            var late = 0
            var hours = 0.0
            var anyHours = false
            val occ = mutableListOf<Occurrence>()
        }

        val acc = LinkedHashMap<Key, Acc>()
        for (s in inRange) {
            for (e in s.nonPresent) {
                val a = acc.getOrPut(Key(s.classId, e.studentId)) { Acc(s.className, e.name, e.massarCode) }
                a.occ += Occurrence(s.epochDay, s.period, s.teacherName, e.status)
                when (e.status) {
                    Status.ABSENT -> {
                        a.abs++
                        periodHours(s.period)?.let { a.hours += it; a.anyHours = true }
                    }
                    Status.LATE -> a.late++
                    Status.PRESENT -> {}
                }
            }
        }

        val students = acc.map { (k, a) ->
            StudentWeek(
                classId = k.classId, className = a.className, studentId = k.studentId,
                name = a.name, massarCode = a.code, absences = a.abs, lates = a.late,
                absentHours = if (a.anyHours) a.hours else null, occurrences = a.occ.toList()
            )
        }.sortedWith(
            compareByDescending<StudentWeek> { it.absences }
                .thenByDescending { it.lates }
                .thenBy { it.className.lowercase() }
                .thenBy { it.name.lowercase() }
        )

        val classes = inRange.groupBy { it.classId }.map { (cid, list) ->
            ClassWeek(
                classId = cid,
                className = list.first().className,
                sessions = list.size,
                absences = list.sumOf { it.absents },
                lates = list.sumOf { it.lates },
                studentsAbsent = students.count { it.classId == cid && it.absences > 0 }
            )
        }.sortedWith(compareByDescending<ClassWeek> { it.absences }.thenBy { it.className.lowercase() })

        return Result(
            from = from, to = to,
            sessions = inRange.size,
            absences = inRange.sumOf { it.absents },
            lates = inRange.sumOf { it.lates },
            classes = classes,
            students = students
        )
    }
}
