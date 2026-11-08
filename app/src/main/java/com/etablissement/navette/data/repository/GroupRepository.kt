package com.etablissement.navette.data.repository

import com.etablissement.navette.data.model.Group
import com.google.firebase.firestore.FirebaseFirestore
import com.google.firebase.firestore.Query
import com.google.firebase.firestore.ktx.firestore
import com.google.firebase.ktx.Firebase
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.tasks.await

/**
 * Accès aux groupes de navette (collection "groups").
 *
 * Seul le conducteur créateur peut modifier/supprimer son groupe
 * (voir les règles Firestore du README).
 */
class GroupRepository(
    private val db: FirebaseFirestore = Firebase.firestore
) {
    private val collection get() = db.collection(COLLECTION)

    /** Flux temps réel de tous les groupes, triés par nom. */
    fun observeGroups(): Flow<List<Group>> = callbackFlow {
        val registration = collection
            .orderBy("name", Query.Direction.ASCENDING)
            .addSnapshotListener { snapshot, error ->
                if (error != null) {
                    close(error)
                    return@addSnapshotListener
                }
                val groups = snapshot?.documents
                    ?.mapNotNull { it.toObject(Group::class.java) }
                    ?: emptyList()
                trySend(groups)
            }
        awaitClose { registration.remove() }
    }

    /** Crée (id vide) ou met à jour un groupe. Retourne l'id du document. */
    suspend fun save(group: Group): String {
        return if (group.id.isBlank()) {
            collection.add(group.toMap()).await().id
        } else {
            collection.document(group.id).set(group.toMap()).await()
            group.id
        }
    }

    /** Supprime un groupe. */
    suspend fun delete(groupId: String) {
        if (groupId.isNotBlank()) {
            collection.document(groupId).delete().await()
        }
    }

    private fun Group.toMap(): Map<String, Any> = mapOf(
        "name" to name,
        "driverId" to driverId,
        "driverName" to driverName,
        "period" to period,
        "memberIds" to memberIds
    )

    companion object {
        private const val COLLECTION = "groups"
    }
}
