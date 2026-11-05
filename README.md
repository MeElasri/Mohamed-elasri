# Navette Profs 🚍

Application Android de **gestion de la navette des professeurs** d'un établissement.

L'application indique **automatiquement** les professeurs attendus sur la navette,
selon le **jour de la semaine** et l'**heure** actuelle :

- Avant **12h00** → navette du **matin**
- À partir de **12h00** → navette du **soir**

L'administrateur gère la liste des professeurs et leur **planning hebdomadaire**
(pour chaque jour : navette du matin et/ou du soir). Les données sont stockées
et synchronisées via **Firebase Firestore**.

---

## 🧱 Technologies

- **Kotlin** + **Jetpack Compose** (Material 3)
- **Firebase Firestore** (stockage cloud, temps réel)
- Architecture **MVVM** (ViewModel + StateFlow + Repository)

## 📂 Structure

```
app/src/main/java/com/etablissement/navette/
├── NavetteApp.kt                 # Classe Application
├── MainActivity.kt               # Navigation (2 onglets + écran d'édition)
├── data/
│   ├── model/Models.kt           # Teacher, Period (Matin/Soir), Weekday
│   └── repository/TeacherRepository.kt   # Accès Firestore (temps réel)
├── ui/
│   ├── NavetteViewModel.kt       # Logique + état
│   ├── theme/Theme.kt
│   └── screens/
│       ├── HomeScreen.kt         # Navette en cours (automatique)
│       ├── TeachersScreen.kt     # Liste + suppression
│       ├── AddEditTeacherScreen.kt  # Formulaire + planning
│       └── CommonComponents.kt
└── util/ScheduleLogic.kt         # Détermination automatique matin/soir
```

## 🔥 Configuration Firebase (indispensable)

L'application a besoin d'un projet Firebase. Suivez ces étapes :

1. Créez un projet sur [Firebase Console](https://console.firebase.google.com/).
2. Ajoutez une **application Android** avec le nom de package exact :
   **`com.etablissement.navette`**
3. Téléchargez le fichier **`google-services.json`** généré par Firebase.
4. Placez-le dans le dossier **`app/`** (soit `app/google-services.json`).
   - Un modèle `app/google-services.json.template` est fourni comme référence.
5. Dans la console Firebase, ouvrez **Firestore Database** → **Créer une base de données**.
6. Pendant le développement, vous pouvez démarrer en **mode test** (règles ouvertes).
   Pour la production, sécurisez l'accès (voir ci-dessous).

> ⚠️ Le fichier `app/google-services.json` est volontairement ignoré par git
> (voir `.gitignore`) car il contient des identifiants propres à votre projet.

### Exemple de règles Firestore (production minimale)

```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /teachers/{doc} {
      // À restreindre selon votre authentification (admin uniquement)
      allow read, write: if true;
    }
  }
}
```

## ▶️ Compilation & exécution

1. Ouvrez le projet dans **Android Studio** (version récente, Giraffe ou plus).
2. Laissez Android Studio synchroniser Gradle et générer le wrapper si besoin.
3. Assurez-vous que `app/google-services.json` est présent.
4. Lancez l'application sur un émulateur ou un appareil (Android 8.0 / API 26 minimum).

En ligne de commande (si le wrapper Gradle est présent) :

```bash
./gradlew assembleDebug
```

## 🧩 Modèle de données Firestore

Collection **`teachers`**, un document par professeur :

```json
{
  "name": "Mme Benali",
  "subject": "Mathématiques",
  "phone": "0612345678",
  "schedule": {
    "MONDAY":  ["MORNING"],
    "TUESDAY": ["MORNING", "EVENING"],
    "FRIDAY":  ["EVENING"]
  }
}
```

- `schedule` : pour chaque jour (`MONDAY` … `SUNDAY`), la liste des périodes
  (`MORNING`, `EVENING`) où le professeur prend la navette.

## 💡 Utilisation

- **Onglet « Navette »** : affiche automatiquement les profs de la navette en cours
  (jour + matin/soir selon l'heure). Boutons pour basculer manuellement matin/soir
  ou consulter un autre jour. Bouton ↻ pour revenir à « maintenant ».
- **Onglet « Professeurs »** : ajouter (＋), modifier ou supprimer un professeur
  et définir son planning hebdomadaire.
