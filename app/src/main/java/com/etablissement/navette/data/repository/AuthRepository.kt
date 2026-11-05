package com.etablissement.navette.data.repository

import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.auth.ktx.auth
import com.google.firebase.ktx.Firebase
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.tasks.await

/**
 * Gestion de l'authentification des enseignants (Firebase Auth, e-mail + mot de passe).
 */
class AuthRepository(
    private val auth: FirebaseAuth = Firebase.auth
) {
    /** UID de l'utilisateur connecté, ou null. */
    val currentUid: String? get() = auth.currentUser?.uid

    /** E-mail de l'utilisateur connecté, ou null. */
    val currentEmail: String? get() = auth.currentUser?.email

    /**
     * Flux de l'UID de l'utilisateur : émet l'UID à la connexion, null à la déconnexion.
     */
    fun authStateFlow(): Flow<String?> = callbackFlow {
        val listener = FirebaseAuth.AuthStateListener { firebaseAuth ->
            trySend(firebaseAuth.currentUser?.uid)
        }
        auth.addAuthStateListener(listener)
        awaitClose { auth.removeAuthStateListener(listener) }
    }

    /** Connexion avec e-mail + mot de passe. */
    suspend fun signIn(email: String, password: String) {
        auth.signInWithEmailAndPassword(email.trim(), password).await()
    }

    /** Création de compte avec e-mail + mot de passe. */
    suspend fun signUp(email: String, password: String) {
        auth.createUserWithEmailAndPassword(email.trim(), password).await()
    }

    /** Déconnexion. */
    fun signOut() {
        auth.signOut()
    }
}
