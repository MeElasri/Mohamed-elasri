package com.etablissement.navette.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.etablissement.navette.data.model.Period
import com.etablissement.navette.data.model.Teacher
import com.etablissement.navette.data.model.Weekday
import com.etablissement.navette.data.repository.TeacherRepository
import com.etablissement.navette.util.ScheduleLogic
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

/**
 * État de l'écran d'accueil (navette en cours).
 */
data class HomeUiState(
    val day: Weekday = ScheduleLogic.currentDay(),
    val period: Period = ScheduleLogic.currentPeriod(),
    /** true si la période a été choisie automatiquement selon l'heure. */
    val isAuto: Boolean = true
)

class NavetteViewModel(
    private val repository: TeacherRepository
) : ViewModel() {

    // Constructeur sans argument requis par la fabrique par défaut de viewModel().
    constructor() : this(TeacherRepository())

    /** Liste complète des professeurs, synchronisée en temps réel avec Firestore. */
    val teachers: StateFlow<List<Teacher>> = repository.observeTeachers()
        .catch { error -> _errorMessage.value = error.message }
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5_000),
            initialValue = emptyList()
        )

    private val _homeState = MutableStateFlow(HomeUiState())
    val homeState: StateFlow<HomeUiState> = _homeState.asStateFlow()

    private val _errorMessage = MutableStateFlow<String?>(null)
    val errorMessage: StateFlow<String?> = _errorMessage.asStateFlow()

    /** Recalcule le jour et la période courante selon l'heure actuelle. */
    fun refreshNow() {
        _homeState.value = HomeUiState(
            day = ScheduleLogic.currentDay(),
            period = ScheduleLogic.currentPeriod(),
            isAuto = true
        )
    }

    /** Permet à l'admin de basculer manuellement entre Matin et Soir. */
    fun selectPeriod(period: Period) {
        _homeState.value = _homeState.value.copy(period = period, isAuto = false)
    }

    /** Change le jour affiché (consultation d'un autre jour de la semaine). */
    fun selectDay(day: Weekday) {
        _homeState.value = _homeState.value.copy(day = day, isAuto = false)
    }

    /** Professeurs attendus sur la navette pour le jour + période sélectionnés. */
    fun teachersOnNavette(all: List<Teacher>, day: Weekday, period: Period): List<Teacher> =
        all.filter { it.isScheduled(day, period) }

    fun saveTeacher(teacher: Teacher) {
        viewModelScope.launch {
            try {
                repository.save(teacher)
            } catch (e: Exception) {
                _errorMessage.value = "Échec de l'enregistrement : ${e.message}"
            }
        }
    }

    fun deleteTeacher(teacherId: String) {
        viewModelScope.launch {
            try {
                repository.delete(teacherId)
            } catch (e: Exception) {
                _errorMessage.value = "Échec de la suppression : ${e.message}"
            }
        }
    }

    fun teacherById(id: String?): Teacher? =
        if (id.isNullOrBlank()) null else teachers.value.firstOrNull { it.id == id }

    fun clearError() {
        _errorMessage.value = null
    }
}
