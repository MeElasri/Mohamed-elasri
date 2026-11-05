# Règles ProGuard pour l'application Navette Profs

# Conserver les modèles de données utilisés par Firestore (sérialisation/désérialisation)
-keepclassmembers class com.etablissement.navette.data.model.** {
    <init>();
    <fields>;
}
-keep class com.etablissement.navette.data.model.** { *; }
