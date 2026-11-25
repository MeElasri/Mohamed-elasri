# Guide d'utilisation — Présence Classe

Application de gestion de l'**appel progressif** dans un établissement scolaire.
Ce guide explique, pas à pas, comment utiliser l'application au quotidien.

Il s'adresse à deux types d'utilisateurs :

- **L'administration** (direction / surveillance générale) : importe les listes Massar, gère les classes et les élèves, crée les comptes des enseignants, consulte et corrige tous les appels.
- **L'enseignant** : fait l'appel de ses classes, consulte son historique et ses rapports, envoie les absences à l'administration.

> La mise en place technique (création du projet Firebase, premier compte administrateur) est décrite une seule fois dans **`FIREBASE_SETUP.md`**. Ce guide-ci suppose que l'application est déjà installée sur le téléphone et que votre compte a été créé.

---

## 1. Premiers pas

### 1.1 Se connecter

1. Ouvrez l'application. L'écran d'accueil s'affiche ; touchez **Commencer**.
2. Saisissez l'**e-mail** et le **mot de passe** communiqués par l'administration.
3. Touchez **Se connecter**.
   - L'icône 👁 à droite du champ permet d'**afficher / masquer** le mot de passe.

Chaque compte garde sa session ouverte : vous n'avez pas à vous reconnecter à chaque fois.

### 1.2 « Accès non autorisé »

Si vous voyez le message **« Accès non autorisé »**, c'est que votre adresse e-mail n'a pas (ou plus) de profil actif dans l'établissement :

- soit votre compte n'a pas encore été créé par l'administration,
- soit il a été **désactivé**.

👉 Contactez l'administration. Tant qu'un profil n'a pas été créé pour vous, l'application ne montre aucune donnée (sécurité).

### 1.3 Mot de passe oublié

Sur l'écran de connexion, saisissez votre e-mail puis touchez **« Mot de passe oublié ? »**.
Un e-mail de réinitialisation vous est envoyé : suivez le lien pour choisir un nouveau mot de passe.

### 1.4 Changer son mot de passe

Depuis l'écran d'accueil (liste des classes) : menu **⋮** (en haut à droite) → **Changer le mot de passe**.
Saisissez le mot de passe actuel, puis deux fois le nouveau (6 caractères minimum).

### 1.5 Se déconnecter

Menu **⋮** → **Se déconnecter**.
Les appels pas encore envoyés **restent sur le téléphone** et partiront à la prochaine connexion de ce compte.

---

## 2. Guide de l'administration

Le compte **administration** voit **toutes les classes** et dispose de fonctions supplémentaires (bouton 📊 Statistiques, menu ⋮ enrichi, boutons ➕ et ✏️/🗑 sur les classes et les élèves).

### 2.1 Importer les listes depuis Massar

C'est la façon la plus rapide de créer toutes les classes d'un coup.

1. Téléchargez depuis **Massar** les listes d'élèves au format **`.xlsx`** (un fichier, ou un fichier par classe).
2. Écran d'accueil → menu **⋮** → **Importer depuis Massar (.xlsx)**.
3. Sélectionnez **une ou plusieurs** listes.
4. Chaque feuille de chaque fichier devient une **classe** :
   - le nom de la classe est lu dans la ligne de titre (« Classe : … » / « القسم : … »), sinon le nom de la feuille, sinon le nom du fichier ;
   - les en-têtes **français ou arabes** sont reconnus (`Code Massar / رقم التلميذ`, `Nom / النسب`, `Prénom / الاسم`, `Nom et prénom`) ;
   - les lignes de titre et de pied de page (totaux, signatures…) sont ignorées.
5. Un récapitulatif indique, pour chaque classe, le nombre d'élèves **ajoutés** et les doublons **ignorés**.

> **Réimport** : vous pouvez réimporter la même liste à tout moment — seuls les **nouveaux** élèves sont ajoutés, les élèves déjà présents ne sont pas dupliqués.
>
> **Ancien format `.xls`** : ouvrez-le dans Excel et enregistrez-le en **`.xlsx`** avant l'import.

### 2.2 Gérer les classes

Sur l'écran d'accueil :

- **Créer** : bouton **➕ Nouvelle classe**. Renseignez le nom (ex. `TCS-3`), le niveau / filière (facultatif) et l'année scolaire.
- **Modifier** : icône ✏️ sur la carte de la classe.
- **Supprimer** : icône 🗑. La classe et sa liste d'élèves sont supprimées et la classe est retirée des enseignants concernés.
  - ⚠️ Les **appels déjà enregistrés restent dans l'historique** même après la suppression de la classe.

### 2.3 Gérer les élèves d'une classe

Touchez une classe pour l'ouvrir, puis :

- **Ajouter un par un** : bouton **➕**. Pour une saisie rapide de toute une liste, utilisez **« + Suivant »** : l'élève est enregistré et le formulaire se vide aussitôt pour le suivant.
- **Importer une liste** dans une classe existante : menu **⋮** → **Importer des élèves (Massar / Excel / CSV)**. Formats acceptés : `.xlsx` (Massar ou Excel) et `.csv`. Les doublons sont ignorés.
- **Modifier / supprimer** un élève : icônes ✏️ / 🗑 à droite de son nom.

Le **Code Massar** est facultatif mais recommandé : il figure dans les rapports et sert à éviter les doublons lors des réimports.

### 2.4 Créer les comptes des enseignants et attribuer les classes

1. Écran d'accueil → menu **⋮** → **Enseignants et classes attribuées**.
2. Bouton **Nouveau compte** :
   - **Nom** (ex. `M. Elasri – Informatique`),
   - **E-mail** de l'enseignant,
   - **Mot de passe provisoire** : un mot de passe est **proposé automatiquement** ; vous pouvez le modifier,
   - **Rôle** : Enseignant (ou Administration),
   - **Classes attribuées** : cochez les classes que l'enseignant doit voir.
3. Touchez **Créer**, puis **communiquez à l'enseignant son e-mail et son mot de passe**. Il pourra le changer lui-même (voir §1.3 / §1.4).

**Modifier un enseignant plus tard** : touchez son nom dans la liste.

- Cochez / décochez ses classes → la mise à jour est **immédiate et en direct** sur son téléphone.
- Interrupteur **Compte actif** : désactivez-le pour couper l'accès d'un enseignant sans supprimer son compte (il verra « Accès non autorisé »).

### 2.5 Consulter et corriger les appels

L'administration peut ouvrir **n'importe quelle classe**, consulter son **Historique** et **corriger** un appel (voir §3). Quand l'administration corrige l'appel d'un enseignant, **le nom de l'enseignant d'origine est conservé**.

### 2.6 Absences du jour (toutes les classes)

Pour transmettre un bilan rapide :

- Icône **🔗 Partager** (en haut de l'écran d'accueil) → absences du **jour** pour **toutes les classes**.
- Ou menu **⋮** → **Absences du jour (toutes les classes)**, ou **Absences d'une autre date** pour choisir un jour.

Un message prêt à lire s'ouvre dans le **sélecteur de partage** (WhatsApp, Gmail, Outlook…), avec un **fichier Excel** joint.

**Imprimer** : menu **⋮** → **Imprimer les absences du jour** (ou **Imprimer une autre date**). Le dialogue d'impression d'Android s'ouvre : choisissez une **imprimante** ou **« Enregistrer au format PDF »**. Fonctionne hors connexion.

### 2.7 Statistiques de la semaine et convocations

Écran d'accueil → icône **📊** (ou menu ⋮ → **Statistiques de la semaine**).

- Navigation **semaine précédente / suivante**.
- Totaux de la semaine : nombre d'appels, d'absences, de retards, par classe et par élève.
- **Seuil de convocation** réglable (par défaut 3) : les élèves dont le nombre d'absences **atteint ou dépasse** le seuil sont listés pour **convocation**, avec le détail des séances manquées (jour + créneau) et, quand les créneaux sont lisibles (ex. « 8h-10h »), le **total d'heures** d'absence.
- Bouton de **partage** pour envoyer ce bilan hebdomadaire (message + Excel).
- Bouton **Imprimer** (icône imprimante, en haut) : ouvre le dialogue d'impression Android — **imprimante** ou **PDF**.

---

## 3. Guide de l'enseignant

À la connexion, vous arrivez sur **« Mes classes »** : uniquement les classes que l'administration vous a attribuées. Cette liste se met à jour **en direct** si l'administration modifie vos attributions.

> Vous ne voyez pas une classe ? Elle ne vous est pas encore attribuée → contactez l'administration.
> La liste d'élèves est vide ? Elle sera ajoutée par l'administration.

### 3.1 Faire l'appel

1. Touchez la classe, puis **Faire l'appel**.
2. **Date** : aujourd'hui par défaut ; touchez-la pour en choisir une autre.
3. **Séance** : saisissez le créneau (ex. `8h-10h`) ou touchez une des **suggestions** (8h-10h, 10h-12h, 14h-16h, 16h-18h). Le créneau est facultatif mais utile pour les rapports et les statistiques d'heures.
4. Marquez les élèves :
   - **Touchez la ligne** d'un élève pour basculer **Présent ⇄ Absent** rapidement, ou
   - utilisez les pastilles **P** (Présent), **A** (Absent), **R** (Retard) à droite de chaque nom.
   - En bas de l'écran, un compteur affiche en continu le nombre d'**absents / retards / présents**.
   - **Tous présents** : l'icône ↻ remet toute la classe en « présent ».
5. Touchez **Enregistrer**.

### 3.2 La « séance précédente » (appel progressif)

C'est le cœur de l'application. Dès que vous ouvrez l'appel, un bandeau **« Séance précédente »** apparaît :

- il affiche le dernier appel fait **le même jour pour cette classe** (par n'importe quel collègue) : enseignant, heure, liste des absents et retards ;
- les élèves **absents à la séance précédente sont automatiquement pré-marqués absents** — vous n'avez qu'à corriger ceux qui sont finalement arrivés ;
- bouton **Réappliquer** : réapplique les absences de la séance précédente si vous avez tout modifié.

**Billet d'entrée ?** Un élève absent avant mais **présent maintenant** est **surligné en orange**, avec le rappel « billet d'entrée ? ». Un message résume aussi le nombre d'élèves concernés.

> Si rien ne s'affiche : il n'y a pas encore eu d'autre appel ce jour-là pour cette classe (ou il n'est pas encore arrivé sur votre téléphone).

### 3.3 Éviter les doublons

Si vous avez **déjà fait l'appel** pour cette classe, cette date **et** cette séance, l'application **recharge votre appel existant** au lieu d'en créer un deuxième, et vous le signale. Vous modifiez alors le même appel.

### 3.4 Enregistrer et envoyer à l'administration

Après **Enregistrer**, une fenêtre confirme l'enregistrement et propose **Envoyer** :

- **Envoyer** ouvre le sélecteur de partage (WhatsApp, e-mail…) avec un message prêt à lire (absents, retards, présents) **et un fichier Excel joint**. Votre nom d'enseignant est ajouté automatiquement.
- **Terminer** ferme sans envoyer (l'appel reste bien enregistré).

Vous pouvez aussi, plus tard :

- renvoyer un appel précis depuis l'**Historique** (icône de partage) ;
- envoyer **toutes vos classes d'une journée** depuis « Mes classes » → menu ⋮ → **Envoyer mes absences du jour** (ou **Absences d'une autre date**).

### 3.5 Quitter sans enregistrer

Si vous quittez l'appel avec des modifications non enregistrées, l'application **vous prévient** (« Quitter sans enregistrer ? ») pour éviter toute perte.

### 3.6 Historique

Classe → **Historique** : tous les appels (les ~120 derniers jours), du plus récent au plus ancien, avec le nombre d'absents / retards.

- Touchez un appel pour le **modifier**.
- Icône **imprimante** pour l'imprimer (imprimante ou PDF).
- Icône de **partage** pour le renvoyer.
- Possibilité de **supprimer** un appel.

> Vous ne pouvez modifier / supprimer que **vos propres** appels. Les appels des collègues s'ouvrent en **lecture seule** (cadenas 🔒). L'administration, elle, peut tout corriger.

### 3.7 Rapports

Classe → **Rapports** : bilan des absences sur une **période** (ce mois-ci, 30 jours, année scolaire… ou dates personnalisées).

- Tableau par élève : séances enregistrées, **absences**, **retards**, **taux d'absence** (signalé en rouge à partir de **20 %**).
- **Export CSV récapitulatif** : une ligne par élève (ouvrable dans Excel).
- **Export CSV détaillé** : une ligne par absence/retard (date, séance, enseignant, élève, statut).

---

## 4. Fonctionnement hors connexion

L'application fonctionne **sans Internet** :

- un appel est **enregistré immédiatement sur le téléphone**, même hors connexion ;
- il est **envoyé automatiquement au serveur dès le retour d'Internet** — vous n'avez rien à refaire ;
- les collègues voient alors cet appel (et la « séance précédente » se met à jour) à ce moment-là ;
- dans le bandeau et l'historique, un appel pas encore synchronisé est marqué **« en attente d'envoi »**.

Après l'enregistrement, la fenêtre de confirmation indique clairement si l'appel a été **confirmé par le serveur** (✓) ou seulement **enregistré sur le téléphone** (☁︎, en attente de connexion).

> ⚠️ Certaines actions ont besoin d'Internet au moment où vous les faites : se connecter la première fois, créer un compte enseignant, changer de mot de passe, envoyer un partage (WhatsApp/e-mail).

---

## 5. Questions fréquentes

**Je ne vois aucune classe.**
Enseignant : aucune classe ne vous est encore attribuée → contactez l'administration. Administration : importez d'abord les listes Massar (§2.1).

**Un élève manque dans la liste.**
L'administration peut l'ajouter (§2.3) : les élèves ne sont gérés que par l'administration.

**Je me suis trompé dans un appel déjà envoyé.**
Ouvrez l'**Historique**, touchez l'appel, corrigez, puis **Enregistrez** et renvoyez-le si besoin.

**L'import Massar ne trouve aucun élève.**
Vérifiez que le fichier est bien un **`.xlsx`** (pas un ancien `.xls` : réenregistrez-le en `.xlsx` dans Excel) et qu'il contient une colonne de noms ou de codes Massar. Le menu **⋮ → Formats acceptés** (dans une classe) rappelle les colonnes reconnues.

**Deux enseignants font l'appel de la même classe le même jour.**
C'est prévu : le second voit la **séance précédente** du premier et retrouve les absents pré-marqués. Chaque séance reste un appel distinct (classe + jour + enseignant + créneau).

**Mon compte affiche « Accès non autorisé ».**
Votre profil est absent ou désactivé → contactez l'administration (§1.2).

---

## 6. Avant d'utiliser de vraies données d'élèves

Rappels importants (voir `FIREBASE_SETUP.md` et `PRIVACY_POLICY.md`) :

- accord écrit de la direction ;
- déclaration **CNDP** (loi 09-08) : données d'élèves mineurs (nom, code Massar, absences) hébergées chez Google Cloud (Union européenne) ;
- politique de confidentialité à jour ;
- testez d'abord avec des **noms fictifs**.
