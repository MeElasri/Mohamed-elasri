package com.etablissement.navette.data.repository

import com.etablissement.navette.data.model.Establishment
import com.google.firebase.firestore.FirebaseFirestore
import com.google.firebase.firestore.SetOptions
import com.google.firebase.firestore.ktx.firestore
import com.google.firebase.ktx.Firebase
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.tasks.await

/** Erreur lisible par l'utilisateur lors de la tentative de rejoindre un établissement. */
class JoinException(message: String) : Exception(message)

/**
 * Établissements et codes d'accès.
 *
 * - "establishments/{id}" : nom, ville et code d'accès.
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

    // ---------------------------------------------------------------- Enseignant

    /**
     * Rattache l'enseignant à l'établissement correspondant au code d'accès [rawCode].
     * Le serveur n'accepte l'écriture que si le code correspond bien à cet établissement.
     *
     * @throws JoinException avec un message à afficher à l'utilisateur.
     */
    suspend fun join(uid: String, rawCode: String): Establishment {
        val code = normalizeCode(rawCode)
        if (code.isEmpty()) throw JoinException("Saisissez le code d'accès de votre établissement")

        val establishmentId = codes.document(code).get().await().getString("establishmentId")
            ?: throw JoinException("Code d'accès inconnu")
        val establishment = establishments.document(establishmentId).get().await()
            .toObject(Establishment::class.java)
            ?: throw JoinException("Établissement introuvable")

        try {
            teachers.document(uid).set(
                mapOf(
                    "establishmentId" to establishmentId,
                    "establishmentName" to establishment.label(),
                    "joinCode" to code
                ),
                SetOptions.merge()
            ).await()
        } catch (e: Exception) {
            throw JoinException("Impossible de rejoindre l'établissement : ${e.message}")
        }
        return establishment
    }

    // ------------------------------------------------------------ Administrateur

    /** Crée un établissement et son code d'accès. */
    suspend fun create(name: String, city: String): Establishment {
        val ref = establishments.document()
        val code = generateCode()
        db.runBatch { batch ->
            batch.set(ref, mapOf("name" to name, "city" to city, "code" to code))
            batch.set(codes.document(code), mapOf("establishmentId" to ref.id))
        }.await()
        return Establishment(id = ref.id, name = name, city = city, code = code)
    }

    /**
     * Remplace le code d'accès : l'ancien cesse de fonctionner pour les nouveaux
     * arrivants ; les enseignants déjà rattachés ne sont pas affectés.
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
