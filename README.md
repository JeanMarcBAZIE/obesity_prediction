# Prédiction de l'Obésité - Machine Learning

## Introduction

Selon l’Organisation mondiale de la santé (OMS), 35 % des adultes dans le monde sont atteints d’obésité ou de surpoids. Cette épidémie touche également les enfants, avec 1 enfant sur 5 en surcharge pondérale. Face à ce problème de santé publique, il est crucial de déterminer et de diagnostiquer les niveaux d’obésité des individus pour mieux les accompagner.

## Objectifs

L’objectif principal est de construire un modèle de classification multiclasses capable de prédire le niveau d’obésité d’un individu en fonction de ses habitudes de vie. Les objectifs spécifiques incluent :

1. Entraîner et valider des modèles à l’aide de pipelines d’apprentissage automatique.
2. Déployer les modèles sur une plateforme GitHub, accessible via une URL web.
3. Mettre en place un système de versioning des modèles et des données.

## Matériel et Méthodes

### Matériel

- **Source des données** : Les données proviennent de l’UCI Machine Learning Repository et comprennent des informations sur les habitudes alimentaires et les conditions physiques d’individus du Mexique, du Pérou et de la Colombie.
- **Structure des données** :
  - **Attributs** : 17
  - **Enregistrements** : 2111
  - **Classes** : Niveau d’obésité (poids insuffisant, poids normal, surpoids niveaux I et II, obésité types I, II, III)
- **Préparation des données** : 77 % générées synthétiquement avec SMOTE et 23 % collectées via une plateforme web.

### Méthodes

1. **Préparation des données** :
   - Nettoyage des données pour supprimer les valeurs manquantes ou dupliquées.
   - Extraction des caractéristiques pertinentes.

2. **Modélisation** :
   - Algorithme utilisé : `ExtraTreesClassifier` (via la bibliothèque Scikit-learn).
   - Métrique d’évaluation : Accuracy (données bien équilibrées).

3. **Organisation des données et résultats** :
   - Données originales : `../assets/data_base`
   - Données d’entraînement et de test : `../assets/data`
   - Caractéristiques extraites : `../assets/features`
   - Modèle sauvegardé : `../assets/models`
   - Métriques : `../assets/metrics`

4. **Versioning** :
   - Outil : DVC (Data Version Control).
   - Deux versions du modèle disponibles :
     - **Version 1** : `n_estimators=150` (branche `main` du dépôt).
     - **Version 2** : `n_estimators=100` (tag : `ExtraCl-n_estimators-100`).

## Bibliothèques utilisées

Les bibliothèques Python suivantes sont utilisées dans ce projet :

- **pandas** : Manipulation et analyse des données.
- **numpy** : Calculs numériques.
- **scikit-learn** : Modélisation machine learning (utilisation de `ExtraTreesClassifier`).
- **matplotlib** : Visualisation des données (si applicable).
- **dvc** : Versioning des données et des modèles.
- **joblib** : Sauvegarde et chargement des modèles.
- **os** : Manipulation des chemins de fichiers et des répertoires.

Assurez-vous d'installer les dépendances en exécutant la commande suivante dans votre environnement Python :

```bash
pip install -r requirements.txt
