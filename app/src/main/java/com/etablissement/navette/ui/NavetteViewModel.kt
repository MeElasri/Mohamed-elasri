package com.etablissement.navette.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.etablissement.navette.data.model.Period
import com.etablissement.navette.data.model.Teacher
import com.etablissement.navette.data.model.Weekday
import com.etablissement.navette.data.repository.AuthRepository
import com.etablissement.navette.data.repository.TeacherRepository
import com.etablissement.navette.util.ScheduleLogic
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.catch
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
    private val teacherRepository: TeacherRepository
) : ViewModel() {

    // Constructeur sans argument requis par la fabrique par défaut de viewModel().
    constructor() : this(AuthRepository(), TeacherRepository())

    /** État d'authentification observé en temps réel. */
    val authState: StateFlow<AuthState> = authRepository.authStateFlow()
        .map { uid -> if (uid == null) AuthState.SignedOut else AuthState.SignedIn(uid) }
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5_000),
            initialValue = AuthState.Loading
        )

    /** Liste de tous les enseignants, synchronisée en temps réel. */
    val teachers: StateFlow<List<Teacher>> = teacherRepository.observeTeachers()
        .catch { error -> _errorMessage.value = error.message }
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5_000),
            initialValue = emptyList()
        )

    /** Profil de l'enseignant connecté (null si pas encore créé). */
    val myProfile: StateFlow<Teacher?> = authRepository.authStateFlow()
        .flatMapLatest { uid ->
            if (uid == null) flowOf(null) else teacherRepository.observeMyProfile(uid)
        }
        .catch { error -> _errorMessage.value = error.message }
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5_000),
            initialValue = null
        )

    private val _homeState = MutableStateFlow(HomeUiState())
    val homeState: StateFlow<HomeUiState> = _homeState.asStateFlow()

    private val _errorMessage = MutableStateFlow<String?>(null)
    val errorMessage: StateFlow<String?> = _errorMessage.asStateFlow()

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
