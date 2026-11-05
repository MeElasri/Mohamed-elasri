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
 */
class TeacherRepository(
    private val db: FirebaseFirestore = Firebase.firestore
) {
    private val collection get() = db.collection(COLLECTION)

    /**
     * Flux temps réel de la liste des professeurs, triée par nom.
     * Émet une nouvelle liste à chaque modification côté Firestore.
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

    /** Crée ou met à jour un professeur. Retourne l'id du document. */
    suspend fun save(teacher: Teacher): String {
        return if (teacher.id.isBlank()) {
            val ref = collection.add(teacher.toMap()).await()
            ref.id
        } else {
            collection.document(teacher.id).set(teacher.toMap()).await()
            teacher.id
        }
    }

    /** Supprime un professeur. */
    suspend fun delete(teacherId: String) {
        if (teacherId.isNotBlank()) {
            collection.document(teacherId).delete().await()
        }
    }

    private fun Teacher.toMap(): Map<String, Any> = mapOf(
        "name" to name,
        "subject" to subject,
        "phone" to phone,
        "schedule" to schedule
    )

    companion object {
        private const val COLLECTION = "teachers"
    }
}
