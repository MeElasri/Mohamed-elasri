package com.etablissement.navette.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.etablissement.navette.data.model.Establishment
import com.etablissement.navette.data.model.Group
import com.etablissement.navette.data.model.Period
import com.etablissement.navette.data.model.Teacher
import com.etablissement.navette.data.model.Weekday
import com.etablissement.navette.data.repository.AuthRepository
import com.etablissement.navette.data.repository.EstablishmentRepository
import com.etablissement.navette.data.repository.GroupRepository
import com.etablissement.navette.data.repository.TeacherRepository
import com.etablissement.navette.util.ScheduleLogic
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.catch
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
    private val establishmentRepository: EstablishmentRepository
) : ViewModel() {

    // Constructeur sans argument requis par la fabrique par défaut de viewModel().
    constructor() : this(
        AuthRepository(), TeacherRepository(), GroupRepository(), EstablishmentRepository()
    )

    val currentUid: String? get() = authRepository.currentUid

    /** État d'authentification observé en temps réel. */
    val authState: StateFlow<AuthState> = authRepository.authStateFlow()
        .map { uid -> if (uid == null) AuthState.SignedOut else AuthState.SignedIn(uid) }
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5_000),
            initialValue = AuthState.Loading
        )

    private val _errorMessage = MutableStateFlow<String?>(null)
    val errorMessage: StateFlow<String?> = _errorMessage.asStateFlow()

    /** Profil de l'enseignant connecté (null si pas encore créé). */
    val myProfile: StateFlow<Teacher?> = authRepository.authStateFlow()
        .flatMapLatest { uid ->
            if (uid == null) flowOf(null)
            else teacherRepository.observeMyProfile(uid)
                .catch { error -> _errorMessage.value = error.message; emit(null) }
        }
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5_000),
            initialValue = null
        )

    /** Identifiant de l'établissement de l'enseignant connecté ("" s'il n'en a pas choisi). */
    private val myEstablishmentId: Flow<String> = myProfile
        .map { it?.establishmentId.orEmpty() }
        .distinctUntilChanged()

    /** Enseignants du même établissement, synchronisés en temps réel. */
    val teachers: StateFlow<List<Teacher>> = myEstablishmentId
        .flatMapLatest { id ->
            if (id.isBlank()) flowOf(emptyList())
            else teacherRepository.observeTeachers(id)
                .catch { error -> _errorMessage.value = error.message; emit(emptyList()) }
        }
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5_000),
            initialValue = emptyList()
        )

    /** Groupes de navette du même établissement, synchronisés en temps réel. */
    val groups: StateFlow<List<Group>> = myEstablishmentId
        .flatMapLatest { id ->
            if (id.isBlank()) flowOf(emptyList())
            else groupRepository.observeGroups(id)
                .catch { error -> _errorMessage.value = error.message; emit(emptyList()) }
        }
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5_000),
            initialValue = emptyList()
        )

    /** Liste de tous les établissements (pour le choix dans le profil). */
    val establishments: StateFlow<List<Establishment>> = establishmentRepository.observeAll()
        .catch { error -> _errorMessage.value = error.message }
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5_000),
            initialValue = emptyList()
        )

    private val _homeState = MutableStateFlow(HomeUiState())
    val homeState: StateFlow<HomeUiState> = _homeState.asStateFlow()

    private val _authBusy = MutableStateFlow(false)
    val authBusy: StateFlow<Boolean> = _authBusy.asStateFlow()

    val currentEmail: String? get() = authRepository.currentEmail

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

    /**
     * Ajoute un établissement (ou réutilise celui qui porte déjà ce nom et cette ville),
     * puis le transmet à [onDone].
     */
    fun createEstablishment(name: String, city: String, onDone: (Establishment) -> Unit) {
        val cleanName = name.trim()
        val cleanCity = city.trim()
        val existing = establishments.value.firstOrNull {
            it.name.equals(cleanName, ignoreCase = true) && it.city.equals(cleanCity, ignoreCase = true)
        }
        if (existing != null) {
            onDone(existing)
            return
        }
        viewModelScope.launch {
            try {
                onDone(establishmentRepository.create(cleanName, cleanCity))
            } catch (e: Exception) {
                _errorMessage.value = "Échec de l'ajout de l'établissement : ${e.message}"
            }
        }
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
