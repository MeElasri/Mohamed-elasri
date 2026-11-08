package com.etablissement.navette.data.repository

import com.etablissement.navette.data.model.Teacher
import com.google.firebase.firestore.FirebaseFirestore
import com.google.firebase.firestore.Query
import com.google.firebase.firestore.ktx.firestore
import com.google.firebase.ktx.Firebase
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.tasks.await

/**
 * Accès aux données des professeurs stockées dans Firestore (collection "teachers").
 *
 * Chaque enseignant possède un document dont l'identifiant est son UID Firebase Auth.
 * Il ne peut modifier que son propre document (voir les règles Firestore du README).
 */
class TeacherRepository(
    private val db: FirebaseFirestore = Firebase.firestore
) {
    private val collection get() = db.collection(COLLECTION)

    /**
     * Flux temps réel de la liste de tous les professeurs, triée par nom.
     */
    fun observeTeachers(): Flow<List<Teacher>> = callbackFlow {
        val registration = collection
            .orderBy("name", Query.Direction.ASCENDING)
            .addSnapshotListener { snapshot, error ->
                if (error != null) {
                    close(error)
                    return@addSnapshotListener
                }
                val teachers = snapshot?.documents
                    ?.mapNotNull { it.toObject(Teacher::class.java) }
                    ?: emptyList()
                trySend(teachers)
            }
        awaitClose { registration.remove() }
    }

    /**
     * Flux temps réel du profil de l'enseignant courant (document teachers/{uid}).
     * Émet null tant qu'aucun profil n'a été créé.
     */
    fun observeMyProfile(uid: String): Flow<Teacher?> = callbackFlow {
        val registration = collection.document(uid)
            .addSnapshotListener { snapshot, error ->
                if (error != null) {
                    close(error)
                    return@addSnapshotListener
                }
                trySend(snapshot?.toObject(Teacher::class.java))
            }
        awaitClose { registration.remove() }
    }

    /**
     * Crée ou met à jour le profil de l'enseignant [uid].
     */
    suspend fun saveProfile(uid: String, teacher: Teacher) {
        collection.document(uid).set(teacher.toMap()).await()
    }

    private fun Teacher.toMap(): Map<String, Any> = mapOf(
        "name" to name,
        "subject" to subject,
        "phone" to phone,
        "email" to email,
        "hasCar" to hasCar,
        "available" to available,
        "schedule" to schedule
    )

    companion object {
        private const val COLLECTION = "teachers"
    }
}
