# Mise en place Firebase — Présence Classe v2

Durée : ~20 minutes, une seule fois, par la personne qui gère l'établissement.

## 1. Créer le projet
1. https://console.firebase.google.com → **Ajouter un projet** → nom : `presence-classe` → désactivez Google Analytics → Créer.
2. Page du projet → icône **Android** :
   - Nom du package : **`com.elasri.presence`** (exactement)
   - → **Télécharger `google-services.json`** et copiez-le dans **`PresenceClasse/app/`** (à côté de `build.gradle.kts`).
   - Ignorez les étapes « Ajouter le SDK » (déjà fait dans le projet).

## 2. Authentification
**Build → Authentication → Commencer → E-mail/Mot de passe → Activer → Enregistrer.**
> Laissez « création de compte » activée : l'application en a besoin pour que l'administration crée les comptes des enseignants. Un inconnu qui créerait un compte ne voit rien : sans profil créé par l'administration, les règles de sécurité refusent tout.

## 3. Base de données
**Build → Firestore Database → Créer une base de données** → emplacement **`europe-west1` (Belgique)** → **mode production** → Créer.

## 4. Règles de sécurité
Firestore → onglet **Règles** → remplacez tout le contenu par celui du fichier **`firebase/firestore.rules`** → **Publier**.

## 5. Index (pour l'historique et les rapports)
Firestore → onglet **Index** → **Composite** → **Créer un index** :
- Collection : `sessions`
- Champ 1 : `classId` — Croissant
- Champ 2 : `epochDay` — Décroissant
- Portée : Collection → Créer (prêt en 2–5 min).

## 6. Créer le premier compte administrateur (vous)
1. **Authentication → Utilisateurs → Ajouter un utilisateur** : votre e-mail + un mot de passe. Copiez l'**UID** affiché.
2. **Firestore → Démarrer une collection** → ID : `users` → ID du document : **collez l'UID** → champs :

| Champ | Type | Valeur |
|---|---|---|
| `name` | string | M. Elasri – Direction |
| `email` | string | votre e-mail |
| `role` | string | `admin` |
| `classIds` | array | (vide) |
| `active` | boolean | true |

Ensuite, tout se fait **dans l'application**.

## 6 bis. Activer l'abonnement (licence)

L'application est **payante par abonnement** : sans licence valide, elle affiche « Abonnement requis »
et reste bloquée. Créez la licence dans Firestore :

- **Démarrer une collection** → ID : `config` → ID du document : `license` → champs :

| Champ | Type | Valeur |
|---|---|---|
| `active` | boolean | true |
| `establishment` | string | nom de l'établissement (affiché si bloqué) |
| `plan` | string | `annuel` (facultatif) |
| `expiresAt` | timestamp | fin de la période payée (ex. `2027-09-01 00:00`) |

Pour **renouveler** plus tard : changez `expiresAt`. Pour **suspendre** : mettez `active` à `false`.
Détails complets dans **`ABONNEMENT.md`**.

## 7. Dans l'application (compte admin)
1. Connexion → menu ⋮ → **Importer depuis Massar (.xlsx)** → sélectionnez toutes les listes.
2. Menu ⋮ → **Enseignants et classes attribuées** → **Nouveau compte** pour chaque enseignant (nom, e-mail, mot de passe provisoire proposé) → cochez ses classes.
3. Communiquez à chaque enseignant son e-mail + mot de passe (il peut le changer via « Mot de passe oublié ? »).
4. Pour modifier les classes d'un enseignant plus tard : touchez son nom → cochez/décochez → la liste se met à jour **en direct** sur son téléphone.

## 8. Vérifier les règles (2 minutes, recommandé)
Firestore → Règles → **Terrain de jeu des règles (Rules Playground)** :
| Test | Attendu |
|---|---|
| `get` `/users/<UID admin>` **sans** authentification | ❌ Refusé |
| `get` `/classes/<id d'une classe>` authentifié avec l'UID d'un enseignant **non** affecté à cette classe | ❌ Refusé |
| `get` `/classes/<id>` avec l'UID d'un enseignant affecté | ✅ Autorisé |

## Fonctionnement de l'appel progressif
- Chaque appel = 1 document `sessions/{classe_jour_enseignant_séance}`.
- À l'ouverture de « Faire l'appel », l'application écoute en direct les appels du **même jour pour la même classe** : la **séance précédente** s'affiche (enseignant, heure, absents) et ses absents sont **pré-marqués**.
- Élève absent avant et présent maintenant → surligné en orange (« billet d'entrée ? »).
- **Hors connexion** : l'appel est gardé sur le téléphone et envoyé automatiquement au retour d'Internet ; les collègues le voient à ce moment-là.

## Coût
Plan gratuit (Spark) suffisant : ~1 écriture par appel. Un lycée de 40 classes × 4 séances ≈ 160 écritures/jour (limite gratuite : 20 000/jour ; lectures : 50 000/jour).

## Avant d'utiliser de vraies données d'élèves
- Accord écrit du directeur.
- Déclaration à la **CNDP** (loi 09-08) : traitement de données d'élèves mineurs (nom, code Massar, absences), hébergé chez Google Cloud (Union européenne).
- Mettre à jour la politique de confidentialité (`PRIVACY_POLICY.md`) et le formulaire « Sécurité des données » de Google Play (données collectées : nom et identifiant des élèves, adresse e-mail des enseignants ; chiffrées en transit ; non partagées ; non vendues).
- Tester d'abord avec des noms fictifs.
