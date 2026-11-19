package com.elasri.presence.data

/**
 * Pure logic of the progressive roll call (no Android / Firebase code, unit-testable).
 */
object AttendanceLogic {

    /** Stable id: one document per class + day + teacher + time slot. */
    fun sessionId(classId: String, epochDay: Long, teacherUid: String, period: String): String =
        "${classId}_${epochDay}_${teacherUid}_${slug(period)}"

    fun slug(s: String): String {
        val t = s.trim().lowercase()
        if (t.isEmpty()) return "seance"
        // Non-Latin slot names (e.g. Arabic) must still give distinct ids.
        return t.replace(Regex("[^a-z0-9]+"), "-").trim('-')
            .ifBlank { "p" + Integer.toHexString(t.hashCode()) }
    }

    /**
     * The lesson taught just before, the same day.
     * - New roll call ([editing] = null): the most recent roll call of the day.
     * - Editing a roll call: the most recent one created before it.
     * Ordered by creation time, so correcting the 8h roll call at 10:30 does not make it "the latest".
     */
    fun previousSession(daySessions: List<Session>, editing: Session?): Session? =
        daySessions
            .filter { it.id != editing?.id }
            .filter { editing == null || it.createdAt < editing.createdAt }
            .maxByOrNull { it.createdAt }

    /**
     * Starting marks for a new roll call: students absent in the previous lesson are pre-marked absent,
     * everybody else present (late students of the previous lesson have arrived, so present).
     */
    fun prefill(students: List<Student>, previous: Session?): Map<String, Status> =
        students.associate { s ->
            s.id to if (previous?.marks?.get(s.id) == Status.ABSENT) Status.ABSENT else Status.PRESENT
        }

    /**
     * Students absent in the previous lesson but not absent now:
     * they arrived late in the day → the administration may ask for an entry slip.
     */
    fun returnedStudents(previous: Session?, current: Map<String, Status>): Set<String> {
        if (previous == null) return emptySet()
        return previous.marks.filter { (id, st) ->
            st == Status.ABSENT && current[id] != null && current[id] != Status.ABSENT
        }.keys
    }

    /** Entries stored with the session for students who are not present. */
    fun nonPresent(students: List<Student>, marks: Map<String, Status>): List<MarkEntry> =
        students.mapNotNull { s ->
            val st = marks[s.id] ?: Status.PRESENT
            if (st == Status.PRESENT) null else MarkEntry(s.id, s.name, s.massarCode, st)
        }

    /** Per-student totals over a set of roll calls; most absent first. */
    fun report(students: List<Student>, sessions: List<Session>): List<StudentReport> =
        students.map { s ->
            var abs = 0
            var late = 0
            var recorded = 0
            for (se in sessions) {
                val st = se.marks[s.id] ?: continue
                recorded++
                if (st == Status.ABSENT) abs++
                if (st == Status.LATE) late++
            }
            StudentReport(s.id, s.name, s.massarCode, abs, late, recorded)
        }.sortedWith(compareByDescending<StudentReport> { it.absences }.thenBy { it.name.lowercase() })
}
