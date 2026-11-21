package com.etablissement.navette.data.repository

import com.etablissement.navette.data.model.Subscription
import com.google.firebase.firestore.DocumentSnapshot
import com.google.firebase.firestore.FieldValue
import com.google.firebase.firestore.FirebaseFirestore
import com.google.firebase.firestore.ktx.firestore
import com.google.firebase.ktx.Firebase
import java.time.Instant
import java.time.ZoneId
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.tasks.await

/**
 * Abonnements individuels (collection "subscriptions", un document par compte).
 *
 * L'enseignant ne peut que créer le sien, une seule fois, pour démarrer l'essai ;
 * tout le reste (période payée, suspension) est réservé à l'administrateur.
 * Ces restrictions sont appliquées côté serveur par firestore.rules.
 */
class SubscriptionRepository(
    private val db: FirebaseFirestore = Firebase.firestore
) {
    private val collection get() = db.collection("subscriptions")

    // Pendant l'écriture, la date du serveur n'est pas encore connue : on l'estime.
    private fun DocumentSnapshot.toSubscription(): Subscription? =
        toObject(Subscription::class.java, DocumentSnapshot.ServerTimestampBehavior.ESTIMATE)

    /** Abonnement du compte [uid], en temps réel (null tant qu'il n'existe pas). */
    fun observe(uid: String): Flow<Subscription?> = callbackFlow {
        val registration = collection.document(uid)
            .addSnapshotListener { snapshot, error ->
                if (error != null) {
                    close(error)
                    return@addSnapshotListener
                }
                trySend(snapshot?.toSubscription())
            }
        awaitClose { registration.remove() }
    }

    /**
     * Crée l'abonnement du compte s'il n'existe pas encore, ce qui démarre la
     * période d'essai à la date du serveur. Sans effet s'il existe déjà.
     */
    suspend fun ensure(uid: String, email: String) {
        val ref = collection.document(uid)
        if (ref.get().await().exists()) return
        ref.set(
            mapOf(
                "email" to email,
                "active" to true,
                "paidUntilMillis" to 0L,
                "createdAt" to FieldValue.serverTimestamp()
            )
        ).await()
    }

    // ------------------------------------------------------------ Administrateur

    /** Tous les abonnements, en temps réel. */
    fun observeAll(): Flow<List<Subscription>> = callbackFlow {
        val registration = collection.addSnapshotListener { snapshot, error ->
            if (error != null) {
                close(error)
                return@addSnapshotListener
            }
            trySend(snapshot?.documents?.mapNotNull { it.toSubscription() } ?: emptyList())
        }
        awaitClose { registration.remove() }
    }

    /**
     * Ajoute [months] mois payés, à la suite de la période payée en cours
     * (ou à partir d'aujourd'hui si elle est terminée), et réactive le compte.
     */
    suspend fun addPaidMonths(subscription: Subscription, months: Int) {
        val from = maxOf(subscription.paidUntilMillis, System.currentTimeMillis())
        val until = Instant.ofEpochMilli(from)
            .atZone(ZoneId.systemDefault())
            .plusMonths(months.toLong())
            .toInstant()
            .toEpochMilli()
        collection.document(subscription.id)
            .update(mapOf("paidUntilMillis" to until, "active" to true)).await()
    }

    /** Met fin immédiatement à la période payée (erreur de saisie, remboursement). */
    suspend fun endPaidPeriod(uid: String) {
        collection.document(uid).update("paidUntilMillis", 0L).await()
    }

    /** Suspend ou réactive un compte. */
    suspend fun setActive(uid: String, active: Boolean) {
        collection.document(uid).update("active", active).await()
    }
}
