package com.etablissement.navette.data.repository

import com.etablissement.navette.data.model.Establishment
import com.google.firebase.firestore.FirebaseFirestore
import com.google.firebase.firestore.ktx.firestore
import com.google.firebase.ktx.Firebase
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.tasks.await

/**
 * Accès à la liste des établissements (collection "establishments").
 * Tout enseignant connecté peut la lire et y ajouter son établissement.
 */
class EstablishmentRepository(
    private val db: FirebaseFirestore = Firebase.firestore
) {
    private val collection get() = db.collection(COLLECTION)

    /** Flux temps réel de tous les établissements, triés par nom. */
    fun observeAll(): Flow<List<Establishment>> = callbackFlow {
        val registration = collection.addSnapshotListener { snapshot, error ->
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

    /** Ajoute un établissement et le retourne avec son identifiant. */
    suspend fun create(name: String, city: String): Establishment {
        val ref = collection.add(mapOf("name" to name, "city" to city)).await()
        return Establishment(id = ref.id, name = name, city = city)
    }

    companion object {
        private const val COLLECTION = "establishments"
    }
}
