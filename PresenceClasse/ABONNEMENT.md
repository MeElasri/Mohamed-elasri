# Abonnement — Présence Classe

L'application est **payante par abonnement, par établissement**. L'accès est débloqué par une
**licence** stockée dans le Firestore de l'établissement, dans le document `config/license`.

- La licence est **lue** par l'application, jamais écrite : les règles de sécurité interdisent toute
  écriture depuis un client (`match /config/{doc} { allow write: if false; }`). L'école ne peut donc
  ni la créer, ni la prolonger, ni la modifier depuis l'app.
- Seul **vous**, via la **console Firebase** (ou l'Admin SDK), pouvez créer et renouveler la licence.
- Quand l'abonnement est **expiré, suspendu ou absent**, l'application affiche un écran
  **« Abonnement requis »** et bloque tout l'usage jusqu'au renouvellement. L'accès est rétabli
  **automatiquement et en direct** dès que vous mettez la date à jour.

> Comme c'est **vous** qui gérez le projet Firebase de chaque établissement, ce mécanisme suffit :
> l'école n'a aucun moyen de contourner la date d'expiration.

---

## 1. Créer la licence (à la mise en service d'une école)

Firebase → **Firestore Database** → **Démarrer une collection** (ou **Ajouter un document**) :

- Collection : **`config`**
- ID du document : **`license`**
- Champs :

| Champ | Type | Exemple | Rôle |
|---|---|---|---|
| `active` | boolean | `true` | Interrupteur général (mettre `false` pour suspendre immédiatement). |
| `establishment` | string | `Lycée Ibn Sina – Fès` | Nom affiché sur l'écran de blocage (facultatif). |
| `plan` | string | `annuel` | Libellé indicatif (facultatif). |
| `expiresAt` | **timestamp** | `2027-09-01 00:00` | Fin de la période payée. |

L'abonnement est **valable tant que** `active = true` **et** que la date/heure actuelle est
**antérieure** à `expiresAt`.

### À propos de `expiresAt`
- **Recommandé : type `timestamp`** (la console propose un sélecteur de date). L'accès est coupé à
  l'instant indiqué.
- **Accepté aussi : type `string`** au format `AAAA-MM-JJ` (ex. `2027-09-01`). Dans ce cas la journée
  indiquée est **incluse** (l'accès reste ouvert jusqu'à la fin de ce jour).

---

## 2. Renouveler (après paiement)

Firestore → `config/license` → modifiez **`expiresAt`** avec la nouvelle date de fin, puis
enregistrez. Rien d'autre à faire : les téléphones se débloquent automatiquement à la prochaine
synchronisation (immédiate s'ils sont en ligne ; sinon au retour d'Internet).

## 3. Suspendre immédiatement

Mettez **`active`** à `false` (par ex. en cas d'impayé). L'application bascule aussitôt sur l'écran
« Abonnement requis ». Repassez `active` à `true` pour rétablir l'accès.

## 4. Votre propre établissement / tests

Créez simplement une licence avec une **date `expiresAt` lointaine** (ex. `2100-01-01`). Pour tester
le blocage, mettez une date passée ou `active = false`, puis rétablissez.

---

## Comportement côté application

- Après la connexion, l'app lit `config/license` **avant** d'afficher les classes.
- Si l'abonnement est valide → usage normal.
- Sinon → écran **« Abonnement requis »** (raison affichée : expiré / suspendu / non configuré),
  avec un bouton **Réessayer** (force une relecture) et **Se déconnecter**.
- La vérification repose sur la date : elle fonctionne **hors connexion** à partir de la dernière
  licence synchronisée.

> La date est comparée à l'horloge du téléphone. Un utilisateur qui reculerait manuellement la date
> de son téléphone pourrait prolonger l'accès ; dans un contexte scolaire B2B, où vous maîtrisez le
> backend, ce risque est négligeable. Une vérification par horloge serveur pourra être ajoutée plus
> tard si nécessaire.

## Sécurité (rappel)

Les règles publiées (`firebase/firestore.rules`) contiennent :

```
match /config/{doc} {
  allow read: if signedIn();   // l'app peut lire la licence
  allow write: if false;        // personne ne peut l'écrire depuis un client
}
```

Vérifiez dans le **Terrain de jeu des règles** qu'une écriture sur `config/license` par un compte
enseignant **ou** administrateur est bien **refusée**.
