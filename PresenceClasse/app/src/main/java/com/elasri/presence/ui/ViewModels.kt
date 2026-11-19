package com.elasri.presence.ui

import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.elasri.presence.PresenceApp
import com.elasri.presence.data.AttendanceLogic
import com.elasri.presence.data.ClassImportResult
import com.elasri.presence.data.FirebaseRepo
import com.elasri.presence.data.Role
import com.elasri.presence.data.SchoolClass
import com.elasri.presence.data.Session
import com.elasri.presence.data.Status
import com.elasri.presence.data.Student
import com.elasri.presence.data.StudentReport
import com.elasri.presence.data.UserProfile
import com.elasri.presence.data.WeeklyStats
import com.elasri.presence.util.AdminReport
import com.elasri.presence.util.WeeklyReport
import com.elasri.presence.data.frMessage
import com.elasri.presence.util.Csv
import com.elasri.presence.util.StudentImport
import com.elasri.presence.util.fmtShort
import com.elasri.presence.util.schoolYearStart
import com.elasri.presence.data.AuthState
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.filterIsInstance
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.withTimeoutOrNull
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.filterNotNull
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.time.LocalDate
import java.util.Locale

// ---------- Helpers ----------

@Composable
fun repository(): FirebaseRepo =
    (LocalContext.current.applicationContext as PresenceApp).repository

fun <VM : ViewModel> factoryOf(builder: () -> VM): ViewModelProvider.Factory =
    object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>): T = builder() as T
    }

// ---------- Login ----------

class LoginViewModel(private val repo: FirebaseRepo) : ViewModel() {
    val busy = MutableStateFlow(false)
    val message = MutableStateFlow<String?>(null)

    fun signIn(email: String, password: String) {
        if (email.isBlank() || password.isBlank()) {
            message.value = "Saisissez l'e-mail et le mot de passe."
            return
        }
        viewModelScope.launch {
            busy.value = true
            message.value = null
            try {
                repo.signIn(email, password)
            } catch (e: Exception) {
                message.value = e.frMessage()
            }
            busy.value = false
        }
    }

    fun resetPassword(email: String) {
        if (email.isBlank()) {
            message.value = "Saisissez d'abord votre e-mail."
            return
        }
        viewModelScope.launch {
            message.value = try {
                repo.resetPassword(email)
                "Un e-mail de réinitialisation a été envoyé à $email."
            } catch (e: Exception) {
                e.frMessage()
            }
        }
    }
}

// ---------- Classes (home) ----------

class ClassesViewModel(private val repo: FirebaseRepo) : ViewModel() {
    val authState = repo.authState
    val classes = repo.myClasses().stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), null)

    suspend fun save(c: SchoolClass): String? = try {
        repo.saveClass(c); null
    } catch (e: Exception) {
        e.frMessage()
    }

    suspend fun delete(c: SchoolClass): String? = try {
        repo.deleteClass(c.id); null
    } catch (e: Exception) {
        e.frMessage()
    }

    fun signOut() = repo.signOut()

    /** Returns null on success, or an error message in French. */
    suspend fun changePassword(current: String, new: String, confirm: String): String? = when {
        new.length < 6 -> "Le nouveau mot de passe doit contenir au moins 6 caractères."
        new != confirm -> "Les deux nouveaux mots de passe ne sont pas identiques."
        new == current -> "Le nouveau mot de passe doit être différent de l'actuel."
        else -> try {
            repo.changePassword(current, new); null
        } catch (e: IllegalArgumentException) {
            e.message
        } catch (e: Exception) {
            e.frMessage()
        }
    }

    /** Every sheet of every file becomes a class. Returns per-class results and per-file errors. */
    suspend fun importFiles(files: List<Pair<String, ByteArray>>): Pair<List<ClassImportResult>, List<String>> {
        val errors = mutableListOf<String>()
        val parsed = withContext(Dispatchers.Default) {
            files.flatMap { (name, bytes) ->
                try {
                    StudentImport.parseFile(bytes, name).also {
                        if (it.isEmpty()) errors += "$name : aucun élève trouvé"
                    }
                } catch (e: Exception) {
                    errors += "$name : ${e.message ?: "fichier illisible"}"
                    emptyList()
                }
            }
        }
        val results = try {
            repo.importClasses(parsed)
        } catch (e: Exception) {
            errors += e.frMessage()
            emptyList()
        }
        return results to errors
    }
}

// ---------- Class detail (students) ----------

class ClassDetailViewModel(private val repo: FirebaseRepo, val classId: String) : ViewModel() {
    val isAdmin: Boolean get() = repo.profile?.isAdmin == true
    val schoolClass = repo.classFlow(classId).stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), null)
    val students = repo.studentsFlow(classId).stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyList())

    suspend fun saveStudent(s: Student): String? = try {
        repo.saveStudent(classId, s); null
    } catch (e: Exception) {
        e.frMessage()
    }

    suspend fun deleteStudent(s: Student): String? = try {
        repo.deleteStudent(classId, s.id); null
    } catch (e: Exception) {
        e.frMessage()
    }

    /** Imports a .xlsx (Massar/Excel) or .csv list into this class. Returns (added, skipped). */
    suspend fun importFile(bytes: ByteArray, fileName: String): Pair<Int, Int> {
        val parsed = withContext(Dispatchers.Default) { StudentImport.parseStudents(bytes, fileName) }
        return repo.importStudents(classId, parsed)
    }
}

// ---------- Attendance (progressive roll call) ----------

data class AttendanceUi(
    val loading: Boolean = true,
    val error: String? = null,
    val cls: SchoolClass? = null,
    val myUid: String? = null,
    val date: LocalDate = LocalDate.now(),
    val period: String = "",
    val students: List<Student> = emptyList(),
    val marks: Map<String, Status> = emptyMap(),
    /** The roll call being edited (null = new roll call). */
    val editing: Session? = null,
    /** Lesson taught just before, the same day (by any teacher). */
    val previous: Session? = null,
    /** Students absent in the previous lesson and now present / late. */
    val returned: Set<String> = emptySet(),
    val readOnly: Boolean = false,
    val dirty: Boolean = false,
    val saving: Boolean = false
) {
    fun count(s: Status) = students.count { (marks[it.id] ?: Status.PRESENT) == s }
}

sealed interface AttendanceEvent {
    data class Message(val text: String) : AttendanceEvent
    data class Saved(val sessionId: String, val confirmed: Boolean) : AttendanceEvent
}

@OptIn(ExperimentalCoroutinesApi::class)
class AttendanceViewModel(
    private val repo: FirebaseRepo,
    private val classId: String,
    private val editSessionId: String?
) : ViewModel() {

    private val _ui = MutableStateFlow(AttendanceUi())
    val ui: StateFlow<AttendanceUi> = _ui.asStateFlow()

    private val _events = MutableSharedFlow<AttendanceEvent>(extraBufferCapacity = 8)
    val events: SharedFlow<AttendanceEvent> = _events.asSharedFlow()

    private val day = MutableStateFlow<Long?>(null)
    private var daySessions: List<Session> = emptyList()
    private var announcedOwn: String? = null

    val isEditing: Boolean get() = editSessionId != null

    init {
        viewModelScope.launch {
            try {
                val me = repo.profile ?: withTimeoutOrNull(10_000) {
                    repo.authState.filterIsInstance<AuthState.Ready>().first().profile
                }
                val cls = repo.getClass(classId) ?: error("Classe introuvable ou non autorisée.")
                val students = repo.getStudents(classId)
                val editing = editSessionId?.let { repo.getSession(it) ?: error("Appel introuvable.") }
                val readOnly = editing != null && me != null && !me.isAdmin && editing.teacherUid != me.uid
                _ui.value = AttendanceUi(
                    loading = false,
                    cls = cls,
                    myUid = me?.uid,
                    date = editing?.let { LocalDate.ofEpochDay(it.epochDay) } ?: LocalDate.now(),
                    period = editing?.period.orEmpty(),
                    students = students,
                    marks = students.associate { it.id to (editing?.marks?.get(it.id) ?: Status.PRESENT) },
                    editing = editing,
                    readOnly = readOnly
                )
                day.value = _ui.value.date.toEpochDay()
            } catch (e: Exception) {
                _ui.update { it.copy(loading = false, error = e.frMessage()) }
            }
        }
        // Live roll calls of the same class and day → "previous lesson" banner + pre-filled absences.
        viewModelScope.launch {
            day.filterNotNull()
                .flatMapLatest { repo.daySessionsFlow(classId, it) }
                .collect { list ->
                    daySessions = list
                    recompute()
                }
        }
    }

    private fun recompute() {
        val u = _ui.value
        if (u.loading || u.error != null) return
        var editing = u.editing
        var marks = u.marks

        if (editSessionId == null && u.myUid != null) {
            // New roll call: if I already saved one for this class/day/slot, load it instead of duplicating.
            val ownId = AttendanceLogic.sessionId(classId, u.date.toEpochDay(), u.myUid, u.period.trim())
            val own = daySessions.firstOrNull { it.id == ownId }
            if (own != null && editing?.id != own.id) {
                editing = own
                if (!u.dirty) marks = u.students.associate { it.id to (own.marks[it.id] ?: Status.PRESENT) }
                if (announcedOwn != own.id) {
                    announcedOwn = own.id
                    _events.tryEmit(AttendanceEvent.Message("Vous avez déjà fait l'appel pour cette séance : il a été chargé."))
                }
            } else if (own == null && editing != null) {
                editing = null
            } else if (own != null) {
                editing = own
            }
        } else if (editing != null) {
            val eid = editing.id
            val fresh = daySessions.firstOrNull { it.id == eid }
            if (fresh != null) editing = fresh
        }

        val previous = AttendanceLogic.previousSession(daySessions, editing)
        if (editing == null && !u.dirty) marks = AttendanceLogic.prefill(u.students, previous)

        _ui.value = u.copy(
            editing = editing,
            marks = marks,
            previous = previous,
            returned = AttendanceLogic.returnedStudents(previous, marks)
        )
    }

    fun setDate(d: LocalDate) {
        if (isEditing || _ui.value.readOnly || d == _ui.value.date) return
        daySessions = emptyList()
        _ui.update { it.copy(date = d, editing = null, previous = null, dirty = false) }
        recompute()
        day.value = d.toEpochDay()
    }

    fun setPeriod(p: String) {
        if (isEditing || _ui.value.readOnly) return
        _ui.update { it.copy(period = p) }
        recompute()
    }

    fun setStatus(studentId: String, s: Status) {
        if (_ui.value.readOnly) return
        _ui.update {
            val marks = it.marks + (studentId to s)
            it.copy(marks = marks, dirty = true, returned = AttendanceLogic.returnedStudents(it.previous, marks))
        }
    }

    fun toggle(studentId: String) {
        val cur = _ui.value.marks[studentId] ?: Status.PRESENT
        setStatus(studentId, if (cur == Status.PRESENT) Status.ABSENT else Status.PRESENT)
    }

    fun markAllPresent() {
        if (_ui.value.readOnly) return
        _ui.update { u ->
            val marks = u.students.associate { it.id to Status.PRESENT }
            u.copy(marks = marks, dirty = true, returned = AttendanceLogic.returnedStudents(u.previous, marks))
        }
    }

    /** Re-applies the absences of the previous lesson. */
    fun applyPrevious() {
        if (_ui.value.readOnly) return
        _ui.update { u ->
            val marks = AttendanceLogic.prefill(u.students, u.previous)
            u.copy(marks = marks, dirty = true, returned = AttendanceLogic.returnedStudents(u.previous, marks))
        }
    }

    fun save() {
        val u = _ui.value
        val cls = u.cls ?: return
        if (u.readOnly || u.saving) return
        if (u.students.isEmpty()) {
            _events.tryEmit(AttendanceEvent.Message("Cette classe n'a pas encore d'élèves (à ajouter par l'administration)."))
            return
        }
        // Our own save comes back through the live listener: don't announce it as "already done".
        if (u.editing == null) u.myUid?.let {
            announcedOwn = AttendanceLogic.sessionId(classId, u.date.toEpochDay(), it, u.period.trim())
        }
        viewModelScope.launch {
            _ui.update { it.copy(saving = true) }
            try {
                val (id, confirmed) = repo.saveSession(
                    cls, u.date.toEpochDay(), u.period.trim(), u.students, u.marks, u.editing
                )
                _ui.update { it.copy(dirty = false, saving = false) }
                _events.emit(AttendanceEvent.Saved(id, confirmed))
            } catch (e: Exception) {
                _ui.update { it.copy(saving = false) }
                _events.emit(AttendanceEvent.Message(e.frMessage()))
            }
        }
    }
}

// ---------- Sessions history ----------

class SessionsViewModel(private val repo: FirebaseRepo, classId: String) : ViewModel() {
    val myUid: String? get() = repo.profile?.uid
    val isAdmin: Boolean get() = repo.profile?.isAdmin == true

    val sessions: StateFlow<Result<List<Session>>?> =
        repo.classSessionsFlow(classId, LocalDate.now().minusDays(120).toEpochDay())
            .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), null)
    val schoolClass = repo.classFlow(classId).stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), null)

    fun canEdit(s: Session) = isAdmin || s.teacherUid == myUid

    fun delete(sessionId: String) {
        viewModelScope.launch { runCatching { repo.deleteSession(sessionId) } }
    }
}

// ---------- Reports ----------

data class ReportUi(
    val loading: Boolean = true,
    val error: String? = null,
    val className: String = "",
    val from: LocalDate = schoolYearStart(),
    val to: LocalDate = LocalDate.now(),
    val rows: List<StudentReport> = emptyList(),
    val sessions: List<Session> = emptyList()
) {
    val totalSessions: Int get() = sessions.size
    val totalAbsences: Int get() = rows.sumOf { it.absences }
    val totalLates: Int get() = rows.sumOf { it.lates }
}

fun StudentReport.rateText(): String = String.format(Locale.FRENCH, "%.1f", rate * 100)

class ReportViewModel(private val repo: FirebaseRepo, private val classId: String) : ViewModel() {
    private val _ui = MutableStateFlow(ReportUi())
    val ui: StateFlow<ReportUi> = _ui.asStateFlow()
    private var loadJob: Job? = null

    init {
        loadJob = viewModelScope.launch {
            val name = runCatching { repo.getClass(classId)?.name }.getOrNull().orEmpty()
            _ui.update { it.copy(className = name) }
            reload()
        }
    }

    fun setRange(a: LocalDate, b: LocalDate) {
        val (from, to) = if (a.isAfter(b)) b to a else a to b
        _ui.update { it.copy(from = from, to = to, loading = true) }
        loadJob?.cancel()
        loadJob = viewModelScope.launch { reload() }
    }

    private suspend fun reload() {
        val u = _ui.value
        try {
            val students = repo.getStudents(classId)
            val sessions = repo.sessionsInRange(classId, u.from, u.to)
            _ui.update {
                it.copy(rows = AttendanceLogic.report(students, sessions), sessions = sessions, loading = false, error = null)
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            _ui.update { it.copy(loading = false, error = e.frMessage()) }
        }
    }

    fun summaryCsv(): ByteArray {
        val u = _ui.value
        val header = listOf("Nom", "Code Massar", "Séances enregistrées", "Absences", "Retards", "Taux d'absence (%)")
        val rows = u.rows.map {
            listOf(it.name, it.massarCode, it.recorded.toString(), it.absences.toString(), it.lates.toString(), it.rateText())
        }
        return Csv.build(header, rows)
    }

    fun detailCsv(): ByteArray {
        val u = _ui.value
        val header = listOf("Date", "Séance", "Enseignant", "Nom", "Code Massar", "Statut")
        val rows = u.sessions.sortedWith(compareBy({ it.epochDay }, { it.createdAt })).flatMap { s ->
            s.nonPresent.map {
                listOf(LocalDate.ofEpochDay(s.epochDay).fmtShort(), s.period, s.teacherName, it.name, it.massarCode, it.status.label)
            }
        }
        return Csv.build(header, rows)
    }
}

// ---------- Weekly statistics (administration) ----------

data class StatsUi(
    val loading: Boolean = true,
    val error: String? = null,
    val weekStart: LocalDate = WeeklyStats.weekStart(LocalDate.now()),
    val threshold: Int = 3,
    val result: WeeklyStats.Result? = null
) {
    val weekEnd: LocalDate get() = weekStart.plusDays(6)
    val convoke: List<WeeklyStats.StudentWeek> get() = result?.toConvoke(threshold).orEmpty()
}

class StatsViewModel(private val repo: FirebaseRepo, private val prefs: android.content.SharedPreferences) : ViewModel() {
    private val _ui = MutableStateFlow(StatsUi(threshold = prefs.getInt(KEY_THRESHOLD, 3)))
    val ui: StateFlow<StatsUi> = _ui.asStateFlow()
    private var loadJob: Job? = null

    init { reload() }

    fun previousWeek() = setWeek(_ui.value.weekStart.minusWeeks(1))
    fun nextWeek() = setWeek(_ui.value.weekStart.plusWeeks(1))
    fun setWeek(anyDay: LocalDate) {
        _ui.update { it.copy(weekStart = WeeklyStats.weekStart(anyDay)) }
        reload()
    }

    fun setThreshold(n: Int) {
        val v = n.coerceIn(1, 30)
        prefs.edit().putInt(KEY_THRESHOLD, v).apply()
        _ui.update { it.copy(threshold = v) }
    }

    fun reload() {
        loadJob?.cancel()
        loadJob = viewModelScope.launch {
            _ui.update { it.copy(loading = true, error = null) }
            val u = _ui.value
            try {
                val sessions = repo.sessionsBetween(u.weekStart, u.weekEnd)
                val result = withContext(Dispatchers.Default) { WeeklyStats.compute(u.weekStart, u.weekEnd, sessions) }
                _ui.update { it.copy(loading = false, result = result) }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                _ui.update { it.copy(loading = false, error = e.frMessage()) }
            }
        }
    }

    fun report(): AdminReport.Output? = _ui.value.result?.let { WeeklyReport.build(it, _ui.value.threshold) }

    companion object {
        const val KEY_THRESHOLD = "convoke_threshold"
    }
}

// ---------- Teachers (administration) ----------

class TeachersViewModel(private val repo: FirebaseRepo) : ViewModel() {
    val users = repo.usersFlow().stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), null)
    val classes = repo.allClasses().stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyList())
    val myUid: String? get() = repo.profile?.uid

    suspend fun create(name: String, email: String, password: String, role: Role, classIds: List<String>): String? =
        try {
            repo.createUser(name, email, password, role, classIds); null
        } catch (e: Exception) {
            e.frMessage()
        }

    suspend fun update(u: UserProfile, name: String, role: Role, classIds: List<String>, active: Boolean): String? =
        try {
            repo.updateUser(u.uid, name, role, classIds, active); null
        } catch (e: Exception) {
            e.frMessage()
        }
}
