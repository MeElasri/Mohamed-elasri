package com.etablissement.navette.data.model

import com.google.firebase.firestore.DocumentId

/**
 * Liste des matières proposées dans le formulaire de profil.
 */
object Subjects {
    val ALL: List<String> = listOf(
        "Informatique",
        "Mathématiques",
        "SVT",
        "Physique-Chimie",
        "Éducation islamique",
        "Histoire-Géographie",
        "Arabe",
        "Français",
        "Philosophie",
        "Anglais",
        "Éducation sportive (EPS)"
    )
}

/**
 * Période de la journée pour la navette.
 */
enum class Period(val label: String) {
    MORNING("Matin"),
    EVENING("Soir");

    companion object {
        fun fromName(name: String?): Period? = entries.firstOrNull { it.name == name }
    }
}

/**
 * Jour de la semaine, avec libellé en français.
 * Le `name` (MONDAY, TUESDAY, ...) correspond à java.time.DayOfWeek.name,
 * ce qui permet de faire correspondre facilement le jour courant.
 */
enum class Weekday(val label: String) {
    MONDAY("Lundi"),
    TUESDAY("Mardi"),
    WEDNESDAY("Mercredi"),
    THURSDAY("Jeudi"),
    FRIDAY("Vendredi"),
    SATURDAY("Samedi"),
    SUNDAY("Dimanche");

    companion object {
        fun fromName(name: String?): Weekday? = entries.firstOrNull { it.name == name }
    }
}

/**
 * Période couverte par un groupe de navette.
 */
enum class GroupPeriod(val label: String) {
    MORNING("Matin"),
    EVENING("Soir"),
    BOTH("Matin et soir");

    companion object {
        fun fromName(name: String?): GroupPeriod = entries.firstOrNull { it.name == name } ?: BOTH
    }
}

/**
 * Groupe de navette (covoiturage) créé par un conducteur (enseignant ayant une voiture).
 *
 * - [driverId] : UID du conducteur créateur (seul à pouvoir modifier/supprimer le groupe).
 * - [memberIds] : UID des enseignants membres (passagers). Le conducteur n'y figure pas.
 * - [period] : nom d'une valeur de [GroupPeriod].
 *
 * Les noms/téléphones ne sont pas stockés ici : ils sont résolus depuis la liste
 * des enseignants (toujours à jour).
 */
data class Group(
    @DocumentId
    val id: String = "",
    val name: String = "",
    val driverId: String = "",
    val driverName: String = "",
    val period: String = GroupPeriod.BOTH.name,
    val memberIds: List<String> = emptyList()
)

/**
 * Modèle d'un professeur et de son planning de navette.
 *
 * Le planning est stocké sous forme de Map : clé = nom du jour (Weekday.name),
 * valeur = liste des périodes (Period.name) où le prof prend la navette ce jour-là.
 *
 * Exemple : { "MONDAY": ["MORNING"], "TUESDAY": ["MORNING", "EVENING"] }
 *
 * Les valeurs par défaut sont nécessaires pour la désérialisation Firestore.
 */
data class Teacher(
    @DocumentId
    val id: String = "",
    val name: String = "",
    val subject: String = "",
    val phone: String = "",
    val email: String = "",
    /** true si l'enseignant possède une voiture (conducteur potentiel). */
    val hasCar: Boolean = false,
    val schedule: Map<String, List<String>> = emptyMap()
) {
    /** Indique si le prof prend la navette le [day] pour la [period] donnée. */
    fun isScheduled(day: Weekday, period: Period): Boolean =
        schedule[day.name]?.contains(period.name) == true
}
