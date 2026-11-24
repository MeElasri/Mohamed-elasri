package com.elasri.presence.data

import android.content.Context
import android.util.Log
import com.elasri.presence.util.AdminReport
import com.elasri.presence.util.StudentTable
import com.elasri.presence.util.schoolYearLabel
import com.google.firebase.FirebaseApp
import com.google.firebase.FirebaseNetworkException
import com.google.firebase.FirebaseTooManyRequestsException
import com.google.firebase.auth.EmailAuthProvider
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.auth.FirebaseAuthInvalidCredentialsException
import com.google.firebase.auth.FirebaseAuthInvalidUserException
import com.google.firebase.auth.FirebaseAuthUserCollisionException
import com.google.firebase.auth.FirebaseAuthWeakPasswordException
import com.google.firebase.firestore.DocumentReference
import com.google.firebase.firestore.DocumentSnapshot
import com.google.firebase.firestore.FieldValue
import com.google.firebase.firestore.FirebaseFirestore
import com.google.firebase.firestore.FirebaseFirestoreException
import com.google.firebase.firestore.MetadataChanges
import com.google.firebase.firestore.Query
import com.google.firebase.firestore.QuerySnapshot
import com.google.firebase.firestore.SetOptions
import com.google.firebase.firestore.Source
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.tasks.await
import kotlinx.coroutines.withTimeoutOrNull
import java.time.LocalDate
import java.time.ZoneId

private const val TAG = "FirebaseRepo"

// ---------------- Firestore → Flow helpers ----------------

fun Query.listen(): Flow<QuerySnapshot> = callbackFlow {
    val reg = addSnapshotListener(MetadataChanges.INCLUDE) { snap, err ->
        if (err != null) close(err) else if (snap != null) trySend(snap)
    }
    awaitClose { reg.remove() }
}

fun DocumentReference.listen(): Flow<DocumentSnapshot> = callbackFlow {
    val reg = addSnapshotListener(MetadataChanges.INCLUDE) { snap, err ->
        if (err != null) close(err) else if (snap != null) trySend(snap)
    }
    awaitClose { reg.remove() }
}

/** French, user-readable message for Firebase errors. */
fun Throwable.frMessage(): String = when (this) {
    is FirebaseAuthInvalidUserException -> "Compte introuvable ou désactivé."
    is FirebaseAuthInvalidCredentialsException -> "E-mail ou mot de passe incorrect."
    is FirebaseAuthWeakPasswordException -> "Mot de passe trop faible (6 caractères minimum)."
    is FirebaseAuthUserCollisionException -> "Un compte existe déjà avec cet e-mail."
    is FirebaseNetworkException -> "Pas de connexion Internet (nécessaire pour cette opération)."
    is FirebaseTooManyRequestsException -> "Trop de tentatives. Réessayez dans quelques minutes."
    is FirebaseFirestoreException -> when (code) {
        FirebaseFirestoreException.Code.PERMISSION_DENIED -> "Accès refusé : droits insuffisants."
        FirebaseFirestoreException.Code.UNAVAILABLE -> "Serveur injoignable : vérifiez la connexion."
        FirebaseFirestoreException.Code.FAILED_PRECONDITION ->
            "Index Firestore manquant : voir FIREBASE_SETUP.md, étape 5."
        else -> message ?: "Erreur Firestore."
    }
    else -> message ?: "Erreur inconnue."
}

/**
 * All shared data lives in Firestore (works offline: changes are kept on the phone and sent later).
 *
 *  users/{uid}                     name, email, role (admin|teacher), classIds[], active
 *  classes/{classId}               name, level, year
 *  classes/{classId}/students/{id} name, massarCode      (id = Massar code when known)
 *  sessions/{classId_day_uid_slot} one roll call (see [AttendanceLogic.sessionId])
 */
@OptIn(ExperimentalCoroutinesApi::class)
class FirebaseRepo(private val context: Context) {

    private val auth = FirebaseAuth.getInstance()
    private val db = FirebaseFirestore.getInstance()
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)

    private val users get() = db.collection("users")
    private val classes get() = db.collection("classes")
    private val sessions get() = db.collection("sessions")
    private val config get() = db.collection("config")
    private fun students(classId: String) = classes.document(classId).collection("students")

    // =====================================================================
    // Authentication & profile
    // =====================================================================

    private val authUid: Flow<String?> = callbackFlow {
        val l = FirebaseAuth.AuthStateListener { trySend(it.currentUser?.uid) }
        auth.addAuthStateListener(l)
        awaitClose { auth.removeAuthStateListener(l) }
    }.distinctUntilChanged()

    val authState: StateFlow<AuthState> = authUid.flatMapLatest { uid ->
        if (uid == null) flowOf(AuthState.SignedOut)
        else users.document(uid).listen().map { snap ->
            val email = auth.currentUser?.email.orEmpty()
            when {
                // first login: nothing in the local cache yet, wait for the server
                !snap.exists() && snap.metadata.isFromCache -> AuthState.Loading
                !snap.exists() -> AuthState.NotAuthorized(email)
                snap.getBoolean("active") == false -> AuthState.NotAuthorized(email)
                else -> AuthState.Ready(parseUser(snap))
            }
        }.catch { e ->
            Log.w(TAG, "profile", e)
            emit(AuthState.NotAuthorized(auth.currentUser?.email.orEmpty()))
        }
    }.stateIn(scope, SharingStarted.Eagerly, AuthState.Loading)

    val profile: UserProfile? get() = (authState.value as? AuthState.Ready)?.profile

    suspend fun signIn(email: String, password: String) {
        auth.signInWithEmailAndPassword(email.trim(), password).await()
    }

    suspend fun resetPassword(email: String) {
        auth.sendPasswordResetEmail(email.trim()).await()
    }

    fun signOut() = auth.signOut()

    /** Firebase requires a recent login to change a password: re-authenticate with the current one first. */
    suspend fun changePassword(current: String, new: String) {
        val user = auth.currentUser ?: error("Non connecté.")
        val email = user.email ?: error("Compte sans e-mail.")
        try {
            user.reauthenticate(EmailAuthProvider.getCredential(email, current)).await()
        } catch (e: FirebaseAuthInvalidCredentialsException) {
            throw IllegalArgumentException("Mot de passe actuel incorrect.")
        }
        user.updatePassword(new).await()
    }

    private fun parseUser(d: DocumentSnapshot) = UserProfile(
        uid = d.id,
        name = d.getString("name").orEmpty(),
        email = d.getString("email").orEmpty(),
        role = Role.of(d.getString("role")),
        classIds = (d.get("classIds") as? List<*>)?.filterIsInstance<String>() ?: emptyList(),
        active = d.getBoolean("active") != false
    )

    // =====================================================================
    // Subscription licence (config/license, written only from the console)
    // =====================================================================

    private fun parseLicense(d: DocumentSnapshot): License {
        // `expiresAt` may be a Firestore timestamp (recommended) or a "YYYY-MM-DD" string.
        val millis = d.getTimestamp("expiresAt")?.toDate()?.time
            ?: d.getString("expiresAt")?.let { s ->
                runCatching {
                    LocalDate.parse(s.trim()).plusDays(1) // valid through the whole expiry day
                        .atStartOfDay(ZoneId.systemDefault()).toInstant().toEpochMilli()
                }.getOrNull()
            }
            ?: 0L
        return License(
            active = d.getBoolean("active") != false,
            establishment = d.getString("establishment").orEmpty(),
            plan = d.getString("plan").orEmpty(),
            expiresAt = millis
        )
    }

    private fun evaluateLicense(snap: DocumentSnapshot): LicenseState = when {
        !snap.exists() && snap.metadata.isFromCache -> LicenseState.Loading // wait for the server
        !snap.exists() -> LicenseState.Blocked(BlockReason.MISSING, null)
        else -> {
            val lic = parseLicense(snap)
            when {
                !lic.active -> LicenseState.Blocked(BlockReason.INACTIVE, lic)
                lic.expiresAt <= 0L -> LicenseState.Blocked(BlockReason.MISSING, lic)
                lic.expiresAt <= System.currentTimeMillis() -> LicenseState.Blocked(BlockReason.EXPIRED, lic)
                else -> LicenseState.Valid(lic)
            }
        }
    }

    // Bumped by refreshLicense() to restart the listener (e.g. after fixing rules / regaining network).
    private val licenseRefresh = MutableStateFlow(0)

    /**
     * Subscription state of the establishment. Only evaluated once signed in (the rules require it).
     * A read error (no network, or rules not yet published) surfaces a clear "Réessayer" screen
     * instead of an endless spinner.
     */
    val licenseState: StateFlow<LicenseState> =
        combine(authState, licenseRefresh) { st, _ -> st }.flatMapLatest { st ->
            if (st !is AuthState.Ready) flowOf(LicenseState.Loading)
            else config.document("license").listen().map(::evaluateLicense)
                .catch { e -> Log.w(TAG, "license", e); emit(LicenseState.Blocked(BlockReason.ERROR, null)) }
        }.stateIn(scope, SharingStarted.Eagerly, LicenseState.Loading)

    /** Restarts the licence check (used by the "Réessayer" button) and asks the server for a fresh copy. */
    suspend fun refreshLicense() {
        licenseRefresh.value += 1
        runCatching { config.document("license").get(Source.SERVER).await() }
    }

    // =====================================================================
    // Classes (admin: all classes, teacher: the classes assigned to them)
    // =====================================================================

    private fun parseClass(d: DocumentSnapshot): SchoolClass? =
        if (!d.exists()) null else SchoolClass(
            id = d.id,
            name = d.getString("name").orEmpty(),
            level = d.getString("level").orEmpty(),
            year = d.getString("year").orEmpty()
        )

    fun allClasses(): Flow<List<SchoolClass>> =
        classes.listen().map { q -> q.documents.mapNotNull(::parseClass).sortedBy { it.name.lowercase() } }
            .catch { e -> Log.w(TAG, "allClasses", e); emit(emptyList()) }

    /** Classes visible to the signed-in user, updated live when the administration changes assignments. */
    fun myClasses(): Flow<List<SchoolClass>> = authState.flatMapLatest { st ->
        val p = (st as? AuthState.Ready)?.profile
        when {
            p == null -> flowOf(emptyList())
            p.isAdmin -> allClasses()
            p.classIds.isEmpty() -> flowOf(emptyList())
            else -> combine(p.classIds.distinct().map { id ->
                classes.document(id).listen().map { parseClass(it) }
                    .catch { e -> Log.w(TAG, "class $id", e); emit(null) }
            }) { arr -> arr.filterNotNull().sortedBy { it.name.lowercase() } }
        }
    }

    fun classFlow(classId: String): Flow<SchoolClass?> =
        classes.document(classId).listen().map { parseClass(it) }.catch { emit(null) }

    suspend fun getClass(classId: String): SchoolClass? = parseClass(classes.document(classId).get().await())

    suspend fun saveClass(c: SchoolClass): String {
        val ref = if (c.id.isBlank()) classes.document() else classes.document(c.id)
        ref.set(mapOf("name" to c.name.trim(), "level" to c.level.trim(), "year" to c.year.trim()), SetOptions.merge())
            .await()
        return ref.id
    }

    /** Deletes the class, its students, and removes it from the teachers' lists. Roll calls are kept. */
    suspend fun deleteClass(classId: String) {
        val studs = students(classId).get().await().documents
        studs.chunked(400).forEach { chunk ->
            db.batch().apply { chunk.forEach { delete(it.reference) } }.commit().await()
        }
        users.whereArrayContains("classIds", classId).get().await().documents.forEach {
            it.reference.update("classIds", FieldValue.arrayRemove(classId)).await()
        }
        classes.document(classId).delete().await()
    }

    // =====================================================================
    // Students (managed by the administration)
    // =====================================================================

    private fun parseStudent(d: DocumentSnapshot) = Student(
        id = d.id,
        name = d.getString("name").orEmpty(),
        massarCode = d.getString("massarCode").orEmpty()
    )

    fun studentsFlow(classId: String): Flow<List<Student>> =
        students(classId).listen()
            .map { q -> q.documents.map(::parseStudent).sortedBy { it.name.lowercase() } }
            .catch { e -> Log.w(TAG, "students", e); emit(emptyList()) }

    suspend fun getStudents(classId: String): List<Student> =
        students(classId).get().await().documents.map(::parseStudent).sortedBy { it.name.lowercase() }

    suspend fun saveStudent(classId: String, s: Student) {
        val code = s.massarCode.trim().uppercase()
        val ref = when {
            s.id.isNotBlank() -> students(classId).document(s.id)
            StudentTable.massarRegex.matches(code) -> students(classId).document(code)
            else -> students(classId).document()
        }
        ref.set(mapOf("name" to s.name.trim(), "massarCode" to code), SetOptions.merge()).await()
    }

    suspend fun deleteStudent(classId: String, studentId: String) {
        students(classId).document(studentId).delete().await()
    }

    /** Returns (added, skipped duplicates). */
    suspend fun importStudents(classId: String, parsed: List<StudentTable.ParsedStudent>): Pair<Int, Int> {
        val existing = getStudents(classId)
        val ids = existing.map { it.id.uppercase() }.toMutableSet()
        val names = existing.map { it.name.trim().lowercase() }.toMutableSet()
        val toWrite = mutableListOf<Pair<DocumentReference, Map<String, Any>>>()
        var skipped = 0
        for (p in parsed) {
            val name = p.name.trim()
            val code = p.code.trim().uppercase()
            val hasCode = StudentTable.massarRegex.matches(code)
            val dup = name.isBlank() || (if (hasCode) code in ids else name.lowercase() in names)
            if (dup) { skipped++; continue }
            val ref = if (hasCode) students(classId).document(code) else students(classId).document()
            toWrite += ref to mapOf("name" to name, "massarCode" to code)
            if (hasCode) ids += code
            names += name.lowercase()
        }
        toWrite.chunked(400).forEach { chunk ->
            db.batch().apply { chunk.forEach { (ref, data) -> set(ref, data) } }.commit().await()
        }
        return toWrite.size to skipped
    }

    /** Creates each class (or reuses the one with the same name) and adds its students. */
    suspend fun importClasses(parsed: List<StudentTable.ParsedClass>): List<ClassImportResult> {
        val existing = classes.get().await().documents.mapNotNull(::parseClass)
            .associateBy { it.name.trim().lowercase() }.toMutableMap()
        return parsed.map { pc ->
            val name = pc.name.trim().ifBlank { "Classe" }
            val found = existing[name.lowercase()]
            val id = found?.id ?: saveClass(SchoolClass(id = "", name = name, year = schoolYearLabel()))
            if (found == null) existing[name.lowercase()] = SchoolClass(id, name)
            val (added, skipped) = importStudents(id, pc.students)
            ClassImportResult(name, found == null, added, skipped)
        }
    }

    // =====================================================================
    // Teachers (administration only)
    // =====================================================================

    fun usersFlow(): Flow<List<UserProfile>> =
        users.listen().map { q -> q.documents.map(::parseUser).sortedWith(compareBy({ it.role }, { it.name.lowercase() })) }
            .catch { e -> Log.w(TAG, "users", e); emit(emptyList()) }

    /**
     * Creates a Firebase account without signing the administrator out:
     * the account is created through a second, temporary FirebaseApp instance.
     */
    suspend fun createUser(name: String, email: String, password: String, role: Role, classIds: List<String>): String {
        val helperName = "account-creator"
        val helperApp = FirebaseApp.getApps(context).firstOrNull { it.name == helperName }
            ?: FirebaseApp.initializeApp(context, FirebaseApp.getInstance().options, helperName)
        val helperAuth = FirebaseAuth.getInstance(helperApp)
        val uid = helperAuth.createUserWithEmailAndPassword(email.trim(), password).await().user?.uid
            ?: error("Création du compte impossible.")
        helperAuth.signOut()
        users.document(uid).set(
            mapOf(
                "name" to name.trim(),
                "email" to email.trim().lowercase(),
                "role" to role.key,
                "classIds" to classIds,
                "active" to true
            )
        ).await()
        return uid
    }

    suspend fun updateUser(uid: String, name: String, role: Role, classIds: List<String>, active: Boolean) {
        users.document(uid).set(
            mapOf("name" to name.trim(), "role" to role.key, "classIds" to classIds, "active" to active),
            SetOptions.merge()
        ).await()
    }

    // =====================================================================
    // Roll calls
    // =====================================================================

    private fun parseSession(d: DocumentSnapshot): Session? {
        if (!d.exists()) return null
        val marks = (d.get("marks") as? Map<*, *>)?.mapNotNull { (k, v) ->
            val id = k as? String ?: return@mapNotNull null
            id to Status.of((v as? Number)?.toInt() ?: 0)
        }?.toMap() ?: emptyMap()
        val np = (d.get("nonPresent") as? List<*>)?.mapNotNull { e ->
            val m = e as? Map<*, *> ?: return@mapNotNull null
            MarkEntry(
                studentId = m["id"] as? String ?: "",
                name = m["name"] as? String ?: "",
                massarCode = m["massarCode"] as? String ?: "",
                status = Status.of((m["status"] as? Number)?.toInt() ?: 1)
            )
        } ?: emptyList()
        return Session(
            id = d.id,
            classId = d.getString("classId").orEmpty(),
            className = d.getString("className").orEmpty(),
            epochDay = d.getLong("epochDay") ?: 0L,
            period = d.getString("period").orEmpty(),
            teacherUid = d.getString("teacherUid").orEmpty(),
            teacherName = d.getString("teacherName").orEmpty(),
            marks = marks,
            nonPresent = np,
            updatedAt = d.getLong("clientUpdatedAt") ?: 0L,
            createdAt = d.getLong("clientCreatedAt") ?: d.getLong("clientUpdatedAt") ?: 0L,
            pending = d.metadata.hasPendingWrites()
        )
    }

    private fun QuerySnapshot.toSessions() = documents.mapNotNull(::parseSession)

    /** All roll calls of a class for one day, live (used for the "previous lesson" banner). */
    fun daySessionsFlow(classId: String, epochDay: Long): Flow<List<Session>> =
        sessions.whereEqualTo("classId", classId).whereEqualTo("epochDay", epochDay).listen()
            .map { it.toSessions() }
            .catch { e -> Log.w(TAG, "daySessions", e); emit(emptyList()) }

    /** Cache first: works offline and right after a save not yet confirmed by the server. */
    suspend fun getSession(id: String): Session? {
        val ref = sessions.document(id)
        val cached = runCatching { ref.get(Source.CACHE).await() }.getOrNull()
        if (cached != null && cached.exists()) return parseSession(cached)
        return try {
            parseSession(ref.get().await())
        } catch (e: FirebaseFirestoreException) {
            // a missing document is reported as PERMISSION_DENIED by the rules (no resource to check)
            if (e.code == FirebaseFirestoreException.Code.PERMISSION_DENIED) null else throw e
        }
    }

    /** Roll calls of a class from [fromDay], newest first (needs the composite index, see README). */
    fun classSessionsFlow(classId: String, fromDay: Long): Flow<Result<List<Session>>> =
        sessions.whereEqualTo("classId", classId)
            .whereGreaterThanOrEqualTo("epochDay", fromDay)
            .orderBy("epochDay", Query.Direction.DESCENDING)
            .listen()
            .map { q -> Result.success(q.toSessions().sortedWith(compareByDescending<Session> { it.epochDay }.thenByDescending { it.updatedAt })) }
            .catch { e -> Log.w(TAG, "classSessions", e); emit(Result.failure(e)) }

    suspend fun sessionsInRange(classId: String, from: LocalDate, to: LocalDate): List<Session> =
        sessions.whereEqualTo("classId", classId)
            .whereGreaterThanOrEqualTo("epochDay", from.toEpochDay())
            .whereLessThanOrEqualTo("epochDay", to.toEpochDay())
            .orderBy("epochDay", Query.Direction.DESCENDING) // same index as the history
            .get().await().toSessions()

    /**
     * Saves a roll call. Works offline: the change is applied on the phone immediately
     * and sent when the connection returns. Returns true if the server confirmed within 3 s.
     */
    suspend fun saveSession(
        cls: SchoolClass,
        epochDay: Long,
        period: String,
        students: List<Student>,
        marks: Map<String, Status>,
        editing: Session?
    ): Pair<String, Boolean> {
        val me = profile ?: error("Non connecté.")
        // When the administration corrects a teacher's roll call, the author stays the teacher.
        val authorUid = editing?.teacherUid ?: me.uid
        val authorName = editing?.teacherName ?: me.name
        val id = editing?.id ?: AttendanceLogic.sessionId(cls.id, epochDay, authorUid, period)
        val nonPresent = AttendanceLogic.nonPresent(students, marks)
        val data = mapOf(
            "classId" to cls.id,
            "className" to cls.name,
            "epochDay" to epochDay,
            "date" to LocalDate.ofEpochDay(epochDay).toString(),
            "period" to period.trim(),
            "teacherUid" to authorUid,
            "teacherName" to authorName,
            "marks" to marks.mapValues { it.value.code },
            "nonPresent" to nonPresent.map {
                mapOf("id" to it.studentId, "name" to it.name, "massarCode" to it.massarCode, "status" to it.status.code)
            },
            "present" to marks.values.count { it == Status.PRESENT },
            "total" to marks.size,
            "clientUpdatedAt" to System.currentTimeMillis(),
            "clientCreatedAt" to (editing?.createdAt?.takeIf { it > 0 } ?: System.currentTimeMillis()),
            "updatedAt" to FieldValue.serverTimestamp()
        )
        val task = sessions.document(id).set(data)
        val confirmed = withTimeoutOrNull(3_000) { task.await(); true } ?: false
        return id to confirmed
    }

    /** Administration: every roll call of the school between two dates (single-field index, no setup needed). */
    suspend fun sessionsBetween(from: LocalDate, to: LocalDate): List<Session> =
        sessions.whereGreaterThanOrEqualTo("epochDay", from.toEpochDay())
            .whereLessThanOrEqualTo("epochDay", to.toEpochDay())
            .get().await().toSessions()

    suspend fun deleteSession(id: String) {
        val task = sessions.document(id).delete()
        withTimeoutOrNull(3_000) { task.await() }
    }

    // =====================================================================
    // Reports sent to the administration (WhatsApp / e-mail)
    // =====================================================================

    private fun Session.toLines() = nonPresent.map {
        AdminReport.Line(className, periodLabel(), it.name, it.massarCode, it.status.label)
    }

    private fun Session.periodLabel() =
        listOf(period, teacherName).filter { it.isNotBlank() }.joinToString(" · ")

    suspend fun adminReportForSession(sessionId: String): AdminReport.Output {
        val s = getSession(sessionId) ?: error("Appel introuvable")
        return AdminReport.forSession(
            s.teacherName, LocalDate.ofEpochDay(s.epochDay),
            AdminReport.CallInfo(s.className, s.periodLabel(), s.present, s.total),
            s.nonPresent.map { AdminReport.Line(s.className, s.periodLabel(), it.name, it.massarCode, it.status.label) }
        )
    }

    /** Administration: every class that day. Teacher: their own roll calls that day. */
    suspend fun adminReportForDay(date: LocalDate): AdminReport.Output {
        val me = profile ?: error("Non connecté.")
        val q = if (me.isAdmin) sessions.whereEqualTo("epochDay", date.toEpochDay())
        else sessions.whereEqualTo("teacherUid", me.uid).whereEqualTo("epochDay", date.toEpochDay())
        val list = q.get().await().toSessions()
        return AdminReport.forDay(
            if (me.isAdmin) "" else me.name, date,
            list.map { AdminReport.CallInfo(it.className, it.periodLabel(), it.present, it.total) },
            list.flatMap { it.toLines() }
        )
    }
}
