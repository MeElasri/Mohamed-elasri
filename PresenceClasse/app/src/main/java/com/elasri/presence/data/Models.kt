package com.elasri.presence.data

enum class Status(val code: Int, val label: String, val short: String) {
    PRESENT(0, "Présent", "P"),
    ABSENT(1, "Absent", "A"),
    LATE(2, "Retard", "R");

    companion object {
        fun of(code: Int): Status = entries.firstOrNull { it.code == code } ?: PRESENT
    }
}

enum class Role(val key: String, val label: String) {
    ADMIN("admin", "Administration"),
    TEACHER("teacher", "Enseignant");

    companion object {
        fun of(key: String?): Role = if (key == ADMIN.key) ADMIN else TEACHER
    }
}

data class UserProfile(
    val uid: String,
    val name: String,
    val email: String,
    val role: Role,
    val classIds: List<String>,
    val active: Boolean = true
) {
    val isAdmin: Boolean get() = role == Role.ADMIN
}

data class SchoolClass(
    val id: String,
    val name: String,
    val level: String = "",
    val year: String = ""
)

data class Student(
    val id: String,
    val name: String,
    val massarCode: String = ""
)

/** A student who was not "present" in a session (kept in the session so reports need no extra reads). */
data class MarkEntry(
    val studentId: String,
    val name: String,
    val massarCode: String,
    val status: Status
)

/** One roll call: one class, one day, one teacher, one time slot. */
data class Session(
    val id: String,
    val classId: String,
    val className: String,
    val epochDay: Long,
    val period: String,
    val teacherUid: String,
    val teacherName: String,
    val marks: Map<String, Status>,
    val nonPresent: List<MarkEntry>,
    val updatedAt: Long,
    /** When the roll call was first saved (kept when it is corrected later): defines lesson order. */
    val createdAt: Long = updatedAt,
    /** true while the roll call is only on this phone (not yet sent to the server). */
    val pending: Boolean = false
) {
    val absents: Int get() = marks.values.count { it == Status.ABSENT }
    val lates: Int get() = marks.values.count { it == Status.LATE }
    val total: Int get() = marks.size
    val present: Int get() = marks.values.count { it == Status.PRESENT }
}

data class StudentReport(
    val studentId: String,
    val name: String,
    val massarCode: String,
    val absences: Int,
    val lates: Int,
    val recorded: Int
) {
    val rate: Float get() = if (recorded == 0) 0f else absences.toFloat() / recorded
}

data class ClassImportResult(val className: String, val isNew: Boolean, val added: Int, val skipped: Int)

/** Login / profile state of the app. */
sealed interface AuthState {
    data object Loading : AuthState
    data object SignedOut : AuthState
    /** Signed in to Firebase, but the administration has not created (or has disabled) the profile. */
    data class NotAuthorized(val email: String) : AuthState
    data class Ready(val profile: UserProfile) : AuthState
}

/**
 * Subscription licence of the establishment, read from `config/license` in Firestore.
 * The document is written only from the Firebase console (security rules forbid any client write),
 * so the school cannot alter or extend it from the app.
 */
data class License(
    val active: Boolean,
    val establishment: String,
    val plan: String,
    /** Expiry as epoch millis (end of the paid period); 0 when no date is set. */
    val expiresAt: Long
)

/** Why the application is locked behind the subscription screen. */
enum class BlockReason { EXPIRED, INACTIVE, MISSING }

/** Subscription state that gates the whole app once the user is signed in. */
sealed interface LicenseState {
    data object Loading : LicenseState
    data class Valid(val license: License) : LicenseState
    data class Blocked(val reason: BlockReason, val license: License?) : LicenseState
}
