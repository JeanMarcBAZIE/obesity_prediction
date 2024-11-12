# Prédiction de l'Obésité - Machine Learning

Ce dépôt contient un Notebook Jupyter pour la prédiction des risques d'obésité en fonction de différents facteurs de style de vie et d'habitudes alimentaires. Ce projet utilise des techniques d'apprentissage automatique pour analyser les données et prédire les catégories d'indice de masse corporelle (IMC) des individus.

## Table des Matières

- [Introduction](#introduction)
- [Jeu de Données](#jeu-de-données)
- [Structure du Projet](#structure-du-projet)
- [Prérequis](#prérequis)
- [Utilisation](#utilisation)
- [Résultats](#résultats)
- [Travail Futur](#travail-futur)
- [Licence](#licence)

## Introduction

L'obésité est une condition de santé influencée par plusieurs facteurs liés au style de vie, tels que l'alimentation, l'activité physique et les habitudes quotidiennes. Ce projet vise à utiliser des algorithmes d'apprentissage automatique pour classer les individus dans des catégories d'IMC (sous-poids, poids normal, surpoids, obésité) en fonction de ces facteurs.

## Jeu de Données

Le jeu de données utilisé contient des informations anonymisées sur le style de vie de différents individus, avec des étiquettes indiquant leur catégorie d'IMC. Les variables incluent des informations sur les habitudes alimentaires, le niveau d'activité physique et d'autres facteurs de style de vie.

## Structure du Projet

- `obesity_prediction.ipynb` : Notebook Jupyter contenant le code pour le prétraitement des données, l'entraînement des modèles et l'évaluation des performances.
- `README.md` : Documentation du projet (ce fichier).
- `data/` : Dossier destiné à stocker le jeu de données.
- `models/` : Dossier pour sauvegarder les modèles entraînés pour une utilisation ou une évaluation ultérieure.

## Prérequis

Ce projet utilise les bibliothèques Python suivantes, que vous pouvez installer avec `pip` :

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
