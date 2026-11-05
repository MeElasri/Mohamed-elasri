package com.etablissement.navette.util

import com.etablissement.navette.data.model.Period
import com.etablissement.navette.data.model.Weekday
import java.time.LocalDateTime
import java.time.LocalTime

/**
 * Logique de détermination automatique de la navette en cours.
 */
object ScheduleLogic {

    /** Heure limite : avant midi => Matin, à partir de midi => Soir. */
    val MORNING_END: LocalTime = LocalTime.NOON

    /** Jour de la semaine courant. */
    fun currentDay(now: LocalDateTime = LocalDateTime.now()): Weekday =
        Weekday.valueOf(now.dayOfWeek.name)

    /**
     * Période courante déterminée automatiquement selon l'heure :
     * avant 12h00 => MATIN, sinon => SOIR.
     */
    fun currentPeriod(now: LocalDateTime = LocalDateTime.now()): Period =
        if (now.toLocalTime().isBefore(MORNING_END)) Period.MORNING else Period.EVENING
}
