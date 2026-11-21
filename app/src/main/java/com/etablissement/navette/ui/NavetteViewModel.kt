package com.etablissement.navette.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.etablissement.navette.data.model.Establishment
import com.etablissement.navette.data.model.Group
import com.etablissement.navette.data.model.Period
import com.etablissement.navette.data.model.Subscription
import com.etablissement.navette.data.model.Teacher
import com.etablissement.navette.data.model.Weekday
import com.etablissement.navette.data.repository.AuthRepository
import com.etablissement.navette.data.repository.EstablishmentRepository
import com.etablissement.navette.data.repository.GroupRepository
import com.etablissement.navette.data.repository.JoinException
import com.etablissement.navette.data.repository.SubscriptionRepository
import com.etablissement.navette.data.repository.TeacherRepository
import com.etablissement.navette.util.ScheduleLogic
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

/** État d'authentification. */
sealed interface AuthState {
    data object Loading : AuthState
    data object SignedOut : AuthState
    data class SignedIn(val uid: String) : AuthState
}

/** État de l'écran navette (jour + période affichés). */
data class HomeUiState(
    val day: Weekday = ScheduleLogic.currentDay(),
    val period: Period = ScheduleLogic.currentPeriod(),
    val isAuto: Boolean = true
)

@OptIn(ExperimentalCoroutinesApi::class)
class NavetteViewModel(
    private val authRepository: AuthRepository,
    private val teacherRepository: TeacherRepository,
    private val groupRepository: GroupRepository,
    private val establishmentRepository: EstablishmentRepository,
    private val subscriptionRepository: SubscriptionRepository
) : ViewModel() {

    // Constructeur sans argument requis par la fabrique par défaut de viewModel().
    constructor() : this(
        AuthRepository(),
        TeacherRepository(),
        GroupRepository(),
        EstablishmentRepository(),
        SubscriptionRepository()
    )

    val currentUid: String? get() = authRepository.currentUid
    val currentEmail: String? get() = authRepository.currentEmail

    private val _errorMessage = MutableStateFlow<String?>(null)
    val errorMessage: StateFlow<String?> = _errorMessage.asStateFlow()

    private val _homeState = MutableStateFlow(HomeUiState())
    val homeState: StateFlow<HomeUiState> = _homeState.asStateFlow()

    private val _joinBusy = MutableStateFlow(false)
    val joinBusy: StateFlow<Boolean> = _joinBusy.asStateFlow()

    private val _authBusy = MutableStateFlow(false)
    val authBusy: StateFlow<Boolean> = _authBusy.asStateFlow()

    private fun <T> Flow<T>.asState(initial: T): StateFlow<T> = stateIn(
        scope = viewModelScope,
        started = SharingStarted.WhileSubscribed(5_000),
        initialValue = initial
    )

    /** État d'authentification observé en temps réel. */
    val authState: StateFlow<AuthState> = authRepository.authStateFlow()
        .map { uid -> if (uid == null) AuthState.SignedOut else AuthState.SignedIn(uid) }
        .asState(AuthState.Loading)

    /** Profil de l'enseignant connecté (null si pas encore créé). */
    val myProfile: StateFlow<Teacher?> = authRepository.authStateFlow()
        .flatMapLatest { uid ->
            if (uid == null) flowOf(null)
            else teacherRepository.observeMyProfile(uid)
                .catch { error -> _errorMessage.value = error.message; emit(null) }
        }
        .asState(null)

    /** Abonnement individuel du compte connecté (essai ou période payée). */
    val mySubscription: StateFlow<Subscription?> = authRepository.authStateFlow()
        .flatMapLatest { uid ->
            if (uid == null) flowOf(null)
            else subscriptionRepository.observe(uid)
                .catch { error -> _errorMessage.value = error.message; emit(null) }
        }
        .asState(null)

    /**
     * Identifiant de l'établissement de l'enseignant si son abonnement lui donne
     * accès, "" sinon : sans accès, aucune donnée n'est demandée au serveur
     * (qui les refuserait de toute façon).
     */
    private val activeEstablishmentId: Flow<String> =
        combine(myProfile, mySubscription) { profile, subscription ->
            val id = profile?.establishmentId.orEmpty()
            if (id.isNotBlank() && subscription?.hasAccess(System.currentTimeMillis()) == true) id
            else ""
        }.distinctUntilChanged()

    /** Enseignants du même établissement, synchronisés en temps réel. */
    val teachers: StateFlow<List<Teacher>> = activeEstablishmentId
        .flatMapLatest { id ->
            if (id.isBlank()) flowOf(emptyList())
            else teacherRepository.observeTeachers(id)
                .catch { error -> _errorMessage.value = error.message; emit(emptyList()) }
        }
        .asState(emptyList())

    /** Groupes de navette du même établissement, synchronisés en temps réel. */
    val groups: StateFlow<List<Group>> = activeEstablishmentId
        .flatMapLatest { id ->
            if (id.isBlank()) flowOf(emptyList())
            else groupRepository.observeGroups(id)
                .catch { error -> _errorMessage.value = error.message; emit(emptyList()) }
        }
        .asState(emptyList())

    /** true si le compte connecté est administrateur. */
    val isAdmin: StateFlow<Boolean> = authRepository.authStateFlow()
        .flatMapLatest { uid ->
            if (uid == null) flowOf(false) else establishmentRepository.observeIsAdmin(uid)
        }
        .asState(false)

    // Les trois flux suivants ne sont observés que par l'écran d'administration.

    /** Tous les établissements. */
    val establishments: StateFlow<List<Establishment>> = establishmentRepository.observeAll()
        .catch { error -> _errorMessage.value = error.message }
        .asState(emptyList())

    /** Tous les abonnements. */
    val allSubscriptions: StateFlow<List<Subscription>> = subscriptionRepository.observeAll()
        .catch { error -> _errorMessage.value = error.message }
        .asState(emptyList())

    /** Tous les enseignants (pour afficher le nom et l'établissement de chaque abonné). */
    val allTeachers: StateFlow<List<Teacher>> = teacherRepository.observeAll()
        .catch { error -> _errorMessage.value = error.message }
        .asState(emptyList())

    init {
        // À chaque connexion : crée l'abonnement du compte s'il n'existe pas encore,
        // ce qui démarre la période d'essai.
        viewModelScope.launch {
            authRepository.authStateFlow().collect { uid ->
                if (uid != null) {
                    try {
                        subscriptionRepository.ensure(uid, authRepository.currentEmail.orEmpty())
                    } catch (e: Exception) {
                        _errorMessage.value = "Impossible d'initialiser l'abonnement : ${e.message}"
                    }
                }
            }
        }
    }

    // --- Navette (écran d'accueil) ---

    fun refreshNow() {
        _homeState.value = HomeUiState(
            day = ScheduleLogic.currentDay(),
            period = ScheduleLogic.currentPeriod(),
            isAuto = true
        )
    }

    fun selectPeriod(period: Period) {
        _homeState.value = _homeState.value.copy(period = period, isAuto = false)
    }

    fun selectDay(day: Weekday) {
        _homeState.value = _homeState.value.copy(day = day, isAuto = false)
    }

    /** Enseignants attendus sur la navette pour le jour + période donnés. */
    fun teachersOnNavette(all: List<Teacher>, day: Weekday, period: Period): List<Teacher> =
        all.filter { it.isScheduled(day, period) }

    // --- Authentification ---

    fun signIn(email: String, password: String) {
        runAuth { authRepository.signIn(email, password) }
    }

    fun signUp(email: String, password: String) {
        runAuth { authRepository.signUp(email, password) }
    }

    fun signOut() {
        authRepository.signOut()
    }

    private fun runAuth(block: suspend () -> Unit) {
        viewModelScope.launch {
            _authBusy.value = true
            try {
                block()
            } catch (e: Exception) {
                _errorMessage.value = translateAuthError(e.message)
            } finally {
                _authBusy.value = false
            }
        }
    }

    // --- Profil ---

    fun saveMyProfile(teacher: Teacher) {
        val uid = authRepository.currentUid ?: return
        viewModelScope.launch {
            try {
                teacherRepository.saveProfile(uid, teacher)
            } catch (e: Exception) {
                _errorMessage.value = "Échec de l'enregistrement : ${e.message}"
            }
        }
    }

    /** Rejoint l'établissement correspondant au code d'accès ; [onJoined] reçoit l'établissement. */
    fun joinEstablishment(code: String, onJoined: (Establishment, String) -> Unit) {
        val uid = authRepository.currentUid ?: return
        viewModelScope.launch {
            _joinBusy.value = true
            try {
                val establishment = establishmentRepository.join(uid, code)
                onJoined(establishment, EstablishmentRepository.normalizeCode(code))
            } catch (e: JoinException) {
                _errorMessage.value = e.message
            } catch (e: Exception) {
                _errorMessage.value = "Impossible de vérifier le code : ${e.message}"
            } finally {
                _joinBusy.value = false
            }
        }
    }

    // --- Administration ---

    private fun adminAction(failure: String, block: suspend () -> Unit) {
        viewModelScope.launch {
            try {
                block()
            } catch (e: Exception) {
                _errorMessage.value = "$failure : ${e.message}"
            }
        }
    }

    /** Ajoute des mois payés à un abonnement (après réception du paiement). */
    fun adminAddPaidMonths(subscription: Subscription, months: Int) =
        adminAction("Échec de l'activation") {
            subscriptionRepository.addPaidMonths(subscription, months)
        }

    fun adminEndPaidPeriod(uid: String) =
        adminAction("Échec de la modification") { subscriptionRepository.endPaidPeriod(uid) }

    fun adminSetSubscriptionActive(uid: String, active: Boolean) =
        adminAction("Échec de la modification") { subscriptionRepository.setActive(uid, active) }

    fun adminCreateEstablishment(name: String, city: String) =
        adminAction("Échec de la création") {
            establishmentRepository.create(name.trim(), city.trim())
        }

    fun adminRegenerateCode(establishment: Establishment) =
        adminAction("Échec du changement de code") {
            establishmentRepository.regenerateCode(establishment)
        }

    // --- Groupes de navette ---

    /** Groupe par son id (depuis la liste en mémoire). */
    fun groupById(id: String?): Group? =
        if (id.isNullOrBlank()) null else groups.value.firstOrNull { it.id == id }

    /** Enseignants correspondant à une liste d'UID (pour l'affichage / le contact). */
    fun teachersByIds(ids: List<String>): List<Teacher> {
        val all = teachers.value
        return ids.mapNotNull { id -> all.firstOrNull { it.id == id } }
    }

    fun saveGroup(group: Group) {
        viewModelScope.launch {
            try {
                groupRepository.save(group)
            } catch (e: Exception) {
                _errorMessage.value = "Échec de l'enregistrement du groupe : ${e.message}"
            }
        }
    }

    fun deleteGroup(groupId: String) {
        viewModelScope.launch {
            try {
                groupRepository.delete(groupId)
            } catch (e: Exception) {
                _errorMessage.value = "Échec de la suppression : ${e.message}"
            }
        }
    }

    fun clearError() {
        _errorMessage.value = null
    }

    private fun translateAuthError(raw: String?): String = when {
        raw == null -> "Erreur d'authentification"
        raw.contains("password is invalid", true) ||
            raw.contains("credential is incorrect", true) -> "E-mail ou mot de passe incorrect"
        raw.contains("no user record", true) -> "Aucun compte pour cet e-mail"
        raw.contains("already in use", true) -> "Cet e-mail est déjà utilisé"
        raw.contains("badly formatted", true) -> "Format d'e-mail invalide"
        raw.contains("at least 6 characters", true) -> "Le mot de passe doit faire au moins 6 caractères"
        else -> raw
    }
}

/**
 * Message à afficher quand l'utilisateur ne peut pas voir les données de son
 * établissement (pas encore rattaché, essai terminé, abonnement expiré ou
 * suspendu) ; null si tout va bien ou si l'abonnement est en cours de chargement.
 */
fun accessProblem(profile: Teacher?, subscription: Subscription?): String? {
    val now = System.currentTimeMillis()
    return when {
        subscription != null && !subscription.active ->
            "Votre abonnement est suspendu. Contactez le fournisseur de l'application."
        subscription != null && !subscription.hasAccess(now) ->
            if (subscription.paidUntilMillis > 0L)
                "Votre abonnement a expiré le ${formatDate(subscription.paidUntilMillis)}. " +
                    "Renouvelez-le pour retrouver la navette et les groupes (onglet « Abonnement »)."
            else
                "Votre période d'essai gratuite est terminée. " +
                    "Abonnez-vous pour continuer (onglet « Abonnement »)."
        profile?.establishmentId.isNullOrBlank() ->
            "Rejoignez d'abord votre établissement dans « Mon profil », avec son code d'accès."
        else -> null
    }
}

/** Date au format jj/mm/aaaa. */
fun formatDate(millis: Long): String =
    java.time.Instant.ofEpochMilli(millis)
        .atZone(java.time.ZoneId.systemDefault())
        .format(java.time.format.DateTimeFormatter.ofPattern("dd/MM/yyyy"))
