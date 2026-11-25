package com.etablissement.navette.data.repository

import com.etablissement.navette.data.model.Establishment
import com.google.firebase.firestore.FirebaseFirestore
import com.google.firebase.firestore.SetOptions
import com.google.firebase.firestore.ktx.firestore
import com.google.firebase.ktx.Firebase
import java.time.Instant
import java.time.ZoneId
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.tasks.await

/** Erreur lisible par l'utilisateur lors de la tentative de rejoindre un établissement. */
class JoinException(message: String) : Exception(message)

/**
 * Établissements et licences.
 *
 * - "establishments/{id}" : nom, ville et licence (active, expiration, places, code).
 * - "licenseCodes/{code}" : renvoie vers l'établissement ; lisible seulement si l'on
 *   connaît le code, ce qui permet à un enseignant de rejoindre sans pouvoir lister.
 * - "admins/{uid}" : comptes administrateurs (créés à la main dans la console Firebase).
 *
 * Toutes les restrictions sont appliquées côté serveur par firestore.rules.
 */
class EstablishmentRepository(
    private val db: FirebaseFirestore = Firebase.firestore
) {
    private val establishments get() = db.collection("establishments")
    private val codes get() = db.collection("licenseCodes")
    private val teachers get() = db.collection("teachers")

    /** Indique en temps réel si l'utilisateur [uid] est administrateur. */
    fun observeIsAdmin(uid: String): Flow<Boolean> = callbackFlow {
        val registration = db.collection("admins").document(uid)
            .addSnapshotListener { snapshot, error ->
                if (error != null) {
                    trySend(false)
                    return@addSnapshotListener
                }
                trySend(snapshot?.exists() == true)
            }
        awaitClose { registration.remove() }
    }

    /** Tous les établissements, triés par nom (administrateur uniquement). */
    fun observeAll(): Flow<List<Establishment>> = callbackFlow {
        val registration = establishments.addSnapshotListener { snapshot, error ->
            if (error != null) {
                close(error)
                return@addSnapshotListener
            }
            val items = snapshot?.documents
                ?.mapNotNull { it.toObject(Establishment::class.java) }
                ?.sortedBy { it.label().lowercase() }
                ?: emptyList()
            trySend(items)
        }
        awaitClose { registration.remove() }
    }

    /** Un établissement et l'état de sa licence, en temps réel. */
    fun observe(id: String): Flow<Establishment?> = callbackFlow {
        val registration = establishments.document(id)
            .addSnapshotListener { snapshot, error ->
                if (error != null) {
                    close(error)
                    return@addSnapshotListener
                }
                trySend(snapshot?.toObject(Establishment::class.java))
            }
        awaitClose { registration.remove() }
    }

    // ---------------------------------------------------------------- Enseignant

    /**
     * Fait rejoindre l'établissement correspondant au code d'activation [rawCode].
     * Le compteur de places et le profil sont mis à jour dans une même transaction,
     * que le serveur n'accepte que si le code, la licence et la limite sont respectés.
     *
     * @throws JoinException avec un message à afficher à l'utilisateur.
     */
    suspend fun join(uid: String, rawCode: String): Establishment {
        val code = normalizeCode(rawCode)
        if (code.isEmpty()) throw JoinException("Saisissez le code d'activation de votre établissement")

        val establishmentId = codes.document(code).get().await().getString("establishmentId")
            ?: throw JoinException("Code d'activation inconnu")
        val ref = establishments.document(establishmentId)
        val teacherRef = teachers.document(uid)

        val current = ref.get().await().toObject(Establishment::class.java)
            ?: throw JoinException("Établissement introuvable")
        val alreadyMember =
            teacherRef.get().await().getString("establishmentId") == establishmentId

        if (!current.active) {
            throw JoinException("La licence de cet établissement est suspendue")
        }
        if (current.expiresAtMillis <= System.currentTimeMillis()) {
            throw JoinException("La licence de cet établissement a expiré")
        }
        if (alreadyMember) return current
        if (current.memberCount >= current.maxTeachers) {
            throw JoinException(
                "Nombre maximum d'enseignants atteint (${current.maxTeachers}) pour cet établissement"
            )
        }

        try {
            return db.runTransaction { tx ->
                val fresh = tx.get(ref).toObject(Establishment::class.java) ?: current
                tx.update(ref, "memberCount", fresh.memberCount + 1)
                tx.set(
                    teacherRef,
                    mapOf(
                        "establishmentId" to establishmentId,
                        "establishmentName" to fresh.label(),
                        "joinCode" to code
                    ),
                    SetOptions.merge()
                )
                fresh
            }.await()
        } catch (e: Exception) {
            throw JoinException(
                "Impossible de rejoindre l'établissement (licence expirée, suspendue ou complète)"
            )
        }
    }

    // ------------------------------------------------------------ Administrateur

    /** Crée un établissement avec sa licence et son code d'activation. */
    suspend fun create(name: String, city: String, maxTeachers: Int, months: Int): Establishment {
        val ref = establishments.document()
        val code = generateCode()
        val establishment = Establishment(
            id = ref.id,
            name = name,
            city = city,
            active = true,
            expiresAtMillis = addMonths(System.currentTimeMillis(), months),
            maxTeachers = maxTeachers,
            memberCount = 0,
            code = code
        )
        db.runBatch { batch ->
            batch.set(
                ref,
                mapOf(
                    "name" to establishment.name,
                    "city" to establishment.city,
                    "active" to establishment.active,
                    "expiresAtMillis" to establishment.expiresAtMillis,
                    "maxTeachers" to establishment.maxTeachers,
                    "memberCount" to establishment.memberCount,
                    "code" to establishment.code
                )
            )
            batch.set(codes.document(code), mapOf("establishmentId" to ref.id))
        }.await()
        return establishment
    }

    /** Suspend ou réactive une licence. */
    suspend fun setActive(id: String, active: Boolean) {
        establishments.document(id).update("active", active).await()
    }

    /** Modifie le nombre maximum d'enseignants. */
    suspend fun setMaxTeachers(id: String, maxTeachers: Int) {
        establishments.document(id).update("maxTeachers", maxTeachers).await()
    }

    /**
     * Prolonge la licence de [months] mois, à partir de sa date de fin
     * (ou d'aujourd'hui si elle est déjà expirée).
     */
    suspend fun extend(establishment: Establishment, months: Int) {
        val from = maxOf(establishment.expiresAtMillis, System.currentTimeMillis())
        establishments.document(establishment.id)
            .update("expiresAtMillis", addMonths(from, months)).await()
    }

    /**
     * Remplace le code d'activation : l'ancien cesse de fonctionner pour les nouveaux
     * arrivants ; les enseignants déjà membres ne sont pas affectés.
     */
    suspend fun regenerateCode(establishment: Establishment): String {
        val newCode = generateCode()
        db.runBatch { batch ->
            if (establishment.code.isNotBlank()) {
                batch.delete(codes.document(establishment.code))
            }
            batch.set(codes.document(newCode), mapOf("establishmentId" to establishment.id))
            batch.update(establishments.document(establishment.id), "code", newCode)
        }.await()
        return newCode
    }

    /** Recalcule le nombre réel d'enseignants rattachés (après des départs, par exemple). */
    suspend fun recount(id: String): Int {
        val count = teachers.whereEqualTo("establishmentId", id).get().await().size()
        establishments.document(id).update("memberCount", count).await()
        return count
    }

    private fun addMonths(fromMillis: Long, months: Int): Long =
        Instant.ofEpochMilli(fromMillis)
            .atZone(ZoneId.systemDefault())
            .plusMonths(months.toLong())
            .toInstant()
            .toEpochMilli()

    companion object {
        // Sans caractères ambigus (0/O, 1/I/L).
        private const val CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
        private const val CODE_LENGTH = 8

        fun generateCode(): String {
            val random = java.security.SecureRandom()
            return (1..CODE_LENGTH)
                .map { CODE_ALPHABET[random.nextInt(CODE_ALPHABET.length)] }
                .joinToString("")
        }

        /** Majuscules, sans espaces ni caractères interdits dans un identifiant. */
        fun normalizeCode(raw: String): String =
            raw.uppercase().filter { it.isLetterOrDigit() }
    }
}
