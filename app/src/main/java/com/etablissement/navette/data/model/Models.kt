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
 * Établissement scolaire et sa licence.
 *
 * Seul un administrateur peut créer un établissement et gérer sa licence.
 * Les enseignants le rejoignent avec le [code] d'accès ; ils n'accèdent aux
 * données que si la licence est [active] et non expirée ([expiresAtMillis]),
 * dans la limite de [maxTeachers] enseignants.
 */
data class Establishment(
    @DocumentId
    val id: String = "",
    val name: String = "",
    val city: String = "",
    /** false = licence suspendue par l'administrateur. */
    val active: Boolean = false,
    /** Date de fin de licence (millisecondes depuis 1970). */
    val expiresAtMillis: Long = 0L,
    /** Nombre maximum d'enseignants autorisés par la licence. */
    val maxTeachers: Int = 0,
    /** Nombre d'enseignants ayant rejoint l'établissement. */
    val memberCount: Int = 0,
    /** Code d'accès à communiquer aux enseignants de l'établissement. */
    val code: String = ""
) {
    /** Libellé affiché : « Nom — Ville » (ou le nom seul). */
    fun label(): String = if (city.isBlank()) name else "$name — $city"

    /** La licence est-elle utilisable à l'instant [now] ? */
    fun hasValidLicense(now: Long): Boolean = active && expiresAtMillis > now
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
    val memberIds: List<String> = emptyList(),
    /** Établissement auquel appartient le groupe (celui du conducteur). */
    val establishmentId: String = ""
) {
    companion object {
        /** Nombre total de places dans la voiture, conducteur compris. */
        const val CAPACITY = 5

        /** Nombre maximum de passagers (hors conducteur). */
        const val MAX_MEMBERS = CAPACITY - 1
    }
}

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
    /** true si l'enseignant est disponible pour la navette (disponible par défaut). */
    val available: Boolean = true,
    /**
     * Jours (Weekday.name) où l'enseignant est indisponible.
     * Liste vide = disponible tous les jours (valeur par défaut).
     */
    val unavailableDays: List<String> = emptyList(),
    /** Établissement de l'enseignant : il ne voit que les collègues et groupes de celui-ci. */
    val establishmentId: String = "",
    val establishmentName: String = "",
    /** Code d'accès utilisé pour rejoindre l'établissement (vérifié par le serveur). */
    val joinCode: String = "",
    val schedule: Map<String, List<String>> = emptyMap()
) {
    /** Indique si l'enseignant est disponible le [day] donné. */
    fun isAvailableOn(day: Weekday): Boolean =
        available && !unavailableDays.contains(day.name)

    /** Indique si le prof prend la navette le [day] pour la [period] donnée. */
    fun isScheduled(day: Weekday, period: Period): Boolean =
        schedule[day.name]?.contains(period.name) == true
}
