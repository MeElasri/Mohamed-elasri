# Présence Classe (v2 — gestion centrale Firebase)

Android app for progressive attendance in a school: the administration imports Massar lists and assigns classes to teachers; each teacher sees only their classes; when a class moves from one teacher to the next, **the new teacher sees the previous lesson's absences live** and they are pre-marked.
Kotlin · Jetpack Compose (Material 3) · Firebase Auth + Cloud Firestore (offline-capable) · minSdk 26 · targetSdk 36.

> **First: follow `FIREBASE_SETUP.md`** (create the Firebase project, add `app/google-services.json`, publish `firebase/firestore.rules`, create the index and the first admin). The app does not build without `google-services.json`.

## Roles
- **Administration** (`role = admin`): import Massar lists, create/edit classes and students, create teacher accounts, tick each teacher's classes, see and correct every roll call, daily report for all classes.
- **Teacher** (`role = teacher`): sees only assigned classes (updated live), takes the roll call, sees the previous lesson banner, edits only their own roll calls, history and reports of their classes.

## Features
- **Classes**: create, edit, delete (name, level/subject, school year).
- **Import from Massar**: on *Mes classes*, tap **Importer Massar** and select one or several `.xlsx` lists downloaded from Massar. Each sheet becomes a class (name read from the "Classe : / القسم :" title line, else the sheet or file name). French or Arabic headers (`Code Massar / رقم التلميذ`, `Nom / النسب`, `Prénom / الاسم`), title rows and footers are handled. Re-importing only adds new students. Old `.xls` must be re-saved as `.xlsx`.
- **Students**: add one by one (with "+ Suivant" for fast entry), edit, delete, or import a `.xlsx`/`.csv` list into an existing class.
- **Attendance call**: pick date + optional slot (8h-10h…), tap a student to mark absent, or choose P / A / R. If a call already exists for that date + slot it is loaded instead of duplicated. Warns before leaving with unsaved changes.
- **Send to the administration**: after saving a call, "Envoyer" opens the share sheet (WhatsApp, Gmail…) with a ready-to-read message (absents, retards, présents) **and an Excel file** attached. Also available per call in *Historique* (share icon) and for **all classes of a day** from *Mes classes* (share icon, or ⋮ → another date). The teacher's name comes from their account.
- **Print**: the **daily report** (*Mes classes* → ⋮ → *Imprimer les absences du jour* / *une autre date*) and the **weekly report** (*Statistiques* → print icon) open the Android print dialog — send to a printer or **save as PDF**. Works offline.
- **History**: every call, with absent/late counts; tap to edit, or delete.
- **Reports**: date range (this month, 30 days, school year…), per-student absences, lates and absence rate (red at ≥ 20 %). Export **summary CSV** or **detailed absence list CSV** (opens in Excel).
- Data is stored in Firestore (EU). Works offline: roll calls are kept on the phone and sent when Internet returns.

## 1. Open and run
1. Install **Android Studio** (latest stable).
2. *File → Open* → select the `PresenceClasse` folder. Wait for Gradle sync (first time downloads ~1 GB).
   - If Studio proposes to upgrade the Android Gradle Plugin or Kotlin, accept.
3. Plug in an Android phone (enable *Developer options → USB debugging*) or create an emulator, then press ▶ Run.

## 2. Before publishing
- **Package name**: `com.elasri.presence` in `app/build.gradle.kts` (`applicationId`). It can never change after the first upload, so change it now if you want another.
- **Version**: increase `versionCode` (1, 2, 3…) and `versionName` for every new upload.

## 3. Build the release bundle (.aab)
*Build → Generate Signed App Bundle or APK → Android App Bundle* → *Create new…* keystore.
- Keep the `.jks` file and its passwords safe (backup outside the PC). With Play App Signing you can request a reset of the upload key if lost, but it's a hassle.
- Output: `app/release/app-release.aab`.

## 4. Publish on Google Play
1. Create a developer account at play.google.com/console (25 USD one-time, ID verification).
2. *Create app* → name "Présence Classe", French, App, Free.
3. **Store listing**: use `store/icon_512.png` and `store/feature_graphic_1024x500.png`; add at least 2 phone screenshots (take them from the emulator).
4. **Privacy policy**: fill in `PRIVACY_POLICY.md`, host it publicly (e.g. GitHub Pages or Google Sites) and paste the URL.
5. **App content**:
   - *Data safety*: data **is collected**: student names + Massar IDs + attendance, staff e-mails; encrypted in transit; not shared; not sold; deletion on request (see FIREBASE_SETUP.md, last section).
   - *Target audience*: **18+** (teachers). Do not select children age groups, otherwise the Families policy applies.
   - *Ads*: No. *Content rating questionnaire*: utility, no sensitive content.
6. **Testing → Closed testing**: upload the `.aab`, add ≥ 12 testers (Google accounts of colleagues), keep them opted in for **14 consecutive days** (required for personal accounts created after Nov 2023).
7. Dashboard → *Apply for production access*, then release to Production.

## Project layout
```
app/src/main/java/com/elasri/presence/
  data/      Models, AttendanceLogic (previous lesson, prefill), FirebaseRepo (Auth + Firestore)
  ui/        Navigation, ViewModels, theme
  ui/screens Classes, ClassDetail, Attendance, Sessions, Report
  util/      CSV import/export, date helpers
store/       Play Store icon + feature graphic
firebase/    firestore.rules, firestore.indexes.json
```

## Next ideas
Arabic translation (move texts to `strings.xml` + `values-ar/`), PDF report, full backup/restore file, per-student history screen.
