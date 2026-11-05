# Navette Profs 🚍

Application Android de **covoiturage / navette entre enseignants** d'un établissement.

Chaque enseignant **crée son compte**, renseigne son **emploi du temps de navette**
(matin et/ou soir, pour chaque jour) et indique s'il **possède une voiture**.
L'application affiche ensuite **automatiquement** les enseignants attendus sur la
navette en cours, en séparant **conducteurs** et **passagers**, afin de faciliter
l'organisation du covoiturage. On peut **contacter** un enseignant en un geste
(Appel, SMS ou WhatsApp).

La navette courante est déterminée automatiquement selon le **jour** et l'**heure** :

- Avant **12h00** → navette du **matin**
- À partir de **12h00** → navette du **soir**

(On peut aussi basculer manuellement matin/soir et consulter un autre jour.)

---

## 🧱 Technologies

- **Kotlin** + **Jetpack Compose** (Material 3)
- **Firebase Authentication** (e-mail + mot de passe)
- **Firebase Firestore** (stockage cloud, temps réel)
- Communication via **intents Android** (téléphone / SMS / WhatsApp)
- Architecture **MVVM** (ViewModel + StateFlow + Repository)

## 📂 Structure

```
app/src/main/java/com/etablissement/navette/
├── NavetteApp.kt                 # Classe Application
├── MainActivity.kt               # Porte d'authentification + navigation (2 onglets)
├── data/
│   ├── model/Models.kt           # Teacher (nom, voiture, planning…), Period, Weekday
│   └── repository/
│       ├── AuthRepository.kt     # Connexion / inscription (Firebase Auth)
│       └── TeacherRepository.kt  # Profils Firestore (teachers/{uid}, temps réel)
├── ui/
│   ├── NavetteViewModel.kt       # État auth + profil + navette
│   ├── theme/Theme.kt
│   └── screens/
│       ├── LoginScreen.kt        # Connexion / création de compte
│       ├── HomeScreen.kt         # Navette en cours : conducteurs / passagers + contact
│       ├── ProfileScreen.kt      # Mon emploi du temps, ma voiture, déconnexion
│       └── CommonComponents.kt
└── util/
    ├── ScheduleLogic.kt          # Détermination automatique matin/soir
    └── ContactActions.kt         # Appel / SMS / WhatsApp
```

## 🔥 Configuration Firebase (indispensable)

1. Créez un projet sur [Firebase Console](https://console.firebase.google.com/).
2. Ajoutez une **application Android** avec le nom de package exact :
   **`com.etablissement.navette`**
3. Téléchargez le fichier **`google-services.json`** et placez-le dans **`app/`**.
   - Un modèle `app/google-services.json.template` est fourni comme référence.
4. **Authentication** → activez le fournisseur **E-mail/Mot de passe**.
5. **Firestore Database** → créez la base de données.

> ⚠️ `app/google-services.json` est volontairement ignoré par git (`.gitignore`)
> car il contient des identifiants propres à votre projet.

### Règles de sécurité Firestore recommandées

Chaque enseignant ne peut modifier **que son propre profil** ; tous les
enseignants connectés peuvent **lire** la liste (pour voir la navette).

```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /teachers/{uid} {
      // Lecture : tout utilisateur connecté
      allow read: if request.auth != null;
      // Écriture : uniquement sur son propre document
      allow write: if request.auth != null && request.auth.uid == uid;
    }
  }
}
```

## ▶️ Compilation & exécution

1. Ouvrez le projet dans **Android Studio** (Giraffe ou plus récent).
2. Laissez Gradle se synchroniser.
3. Vérifiez que `app/google-services.json` est présent.
4. Lancez sur un appareil / émulateur (Android 8.0 / API 26 minimum).

En ligne de commande :

```bash
./gradlew assembleDebug
```

## 🧩 Modèle de données Firestore

Collection **`teachers`**, un document par enseignant, **identifié par son UID** :

```json
{
  "name": "Mme Benali",
  "subject": "Mathématiques",
  "phone": "+2126XXXXXXXX",
  "email": "benali@exemple.ma",
  "hasCar": true,
  "schedule": {
    "MONDAY":  ["MORNING"],
    "TUESDAY": ["MORNING", "EVENING"],
    "FRIDAY":  ["EVENING"]
  }
}
```

- `hasCar` : `true` si l'enseignant a une voiture (conducteur potentiel).
- `schedule` : pour chaque jour (`MONDAY` … `SUNDAY`), les périodes
  (`MORNING`, `EVENING`) où il prend la navette.

## 💡 Utilisation

1. **Connexion / inscription** avec une adresse e-mail.
2. **Onglet « Mon profil »** : renseigner son nom, sa matière, son **téléphone**
   (idéalement au format international pour WhatsApp, ex. `+2126…`), cocher s'il a
   une **voiture**, et définir son **emploi du temps de navette**.
3. **Onglet « Navette »** : voir automatiquement, pour la navette en cours, les
   **conducteurs** (avec voiture) et les **passagers** (sans voiture). Boutons
   **📞 Appeler**, **💬 SMS**, **WhatsApp** pour se contacter et organiser le trajet.

> 💡 Pour que **WhatsApp** fonctionne de façon fiable, saisissez le numéro au
> format international sans espaces (ex. `+212612345678`).
