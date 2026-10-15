# Prédiction du churn client – Telco Customer Churn

Projet de Machine Learning visant à **identifier les clients d'un opérateur télécom susceptibles de résilier leur abonnement et à comprendre les facteurs associés à ce départ.**

**Outils :** Python, pandas, scikit-learn, seaborn, matplotlib, Jupyter

------

## 1. Contexte et objectif

Acquérir un nouveau client coûte généralement plus cher que d'en fidéliser un. Un modèle capable de repérer à l'avance les clients à risque permet de cibler des actions de rétention (appel, offre, changement de contrat).

**Objectifs du projet :**
1. Construire un modèle de classification du churn (`Yes` = le client a résilié).
2. Choisir un modèle et un seuil de décision en fonction d'un objectif métier.
3. Identifier les profils et facteurs associés au départ.

## 2. Données
- Source : jeu de données public *Telco Customer Churn* d'IBM.Le fichier est inclus dans ce dépôt à des fins pédagogiques.
- 7 043 clients, 21 colonnes : démographie, services souscrits, type de contrat, facturation, ancienneté (`tenure`) et cible `Churn`.
- **Classe déséquilibrée** : environ 26,5 % de clients ont résilié.
- **Variables écartées** : customerID (identifiant) et gender (aucune justification métier à son influence sur le départ, et son usage poserait un risque de discrimination).

## 3. Méthodologie

1. **Exploration** : répartition du churn, analyse croisée avec l'ancienneté, le contrat, la charge mensuelle, le type d'internet, le mode de paiement et le support technique.
2. **Nettoyage** : `TotalCharges` converti en numérique. Les 11 valeurs manquantes correspondent à des clients d'ancienneté nulle et ont été remplacées par 0. Aucun doublon.
3. **Séparation train/test** (80 % / 20 %) avec `stratify=y` pour conserver la même proportion de churn dans les deux ensembles.
4. **Prétraitement** avec un `ColumnTransformer`, ajusté **uniquement sur le train** pour éviter toute fuite de données (*data leakage*) :
   - variables numériques : `StandardScaler`
   - variables catégorielles : `OneHotEncoder(handle_unknown="ignore")`
   - 18 variables explicatives → 44 colonnes après encodage.
5. **Modélisation** : régression logistique, Random Forest, équilibrage et rééchantillonnage (sur-échantillonnage, SMOTE, sous-échantillonnage).
6. **Évaluation** : precision, recall, F1-score, AUC, courbe Precision-Recall, choix du seuil.

## 4. Comparaison des modèles (seuil de 0,5)

|Famille          | Modèle                            |AUC train/test| Recall| Precision| F1    |
|-----------------|-----------------------------------|--------------|-------|----------|-------|
|Base             | Régression logistique             | 0,8493/0,8421| 0,5588| 0,6572   | 0,6040|
|Base             | Random Forest                     | 0,9998/0,8240| 0,4893| 0,6079   | 0,5422|
|Pondération      | Régression logistique équilibrée  | 0,8492/0,8418| 0,7834| 0,5026   | 0,6123|
|Rééchantillonnage| Sous-échantillonnage              | 0,8484/0,8429| 0,7861| 0,5095   | 0,6183|
|Rééchantillonnage| SMOTE                             | 0,8486/0,8395| 0,7941| 0,5000   | 0,6136|
|Rééchantillonnage| Sur-échantillonnage               | 0,8490/0,8424| 0,7888| 0,5000   | 0,612 |

**Constats :**
- Le Random Forest sur-apprend : son AUC est de 0,9998 sur l’entraînement contre 0,824 sur le test, alors que la régression logistique reste stable (0,849 contre 0,842). Sur ce jeu de données, un modèle simple et interprétable est le meilleur choix.
- `class_weight="balanced"` ne change pas la capacité de classement du modèle (AUC presque identique) : il déplace seulement le compromis entre precision et recall. Il est plus propre de **régler directement le seuil de décision**.
- Trois stratégies ont été comparées sur le jeu d’entraînement uniquement : sur-échantillonnage aléatoire, SMOTE et sous-échantillonnage aléatoire. Toutes augmentent le recall (d'environ 0,56 à environ 0,79) au prix d’une baisse de la precision, sans améliorer l’AUC (0,84 dans tous les cas). Elles déplacent le compromis precision/recall comme le fait le réglage du seuil, sans gain de performance. Le choix final s’est donc porté sur la régression logistique classique avec un seuil de décision ajusté (0,278), plus simple et plus facile à justifier côté métier.

## 5. Choix du seuil de décision

Par défaut, un client est classé « à risque » si sa probabilité de départ dépasse 0,5. Ce seuil n'est pas optimal quand on veut détecter les départs.

La courbe Precision-Recall (Average Precision ≈ 0,63, pour un taux de churn de 0,265 qui correspond au hasard) montre que le modèle repère réellement les clients à risque.

| Scénario                   | Seuil | Recall| Precision| F1   |
|----------------------------|-------|-------|----------|------|
| Seuil standard             | 0,50  | 0,559 | 0,657    | 0,604|
| Seuil optimal (F1 maximal) | 0,278 | 0,781 | 0,518    | 0,623|

**Interprétation métier :** avec un seuil de 0,278, le modèle détecte environ **78 % des clients qui vont résilier**, mais seulement un client signalé sur deux part réellement.

- Si une campagne de rétention est peu coûteuse et qu'un client perdu coûte cher, un seuil bas (environ 0,25 à 0,30) est pertinent.
- Si le budget est limité, un seuil plus élevé (environ 0,40 à 0,50) cible mieux chaque action.

Le F1-score est très stable entre 0,25 et 0,45 : c'est le coût métier, plus que la métrique, qui doit trancher.

## 6. Facteurs associés au départ

Analyse des coefficients de la régression logistique (odds ratios) :

| Facteur                              | Odds ratio | Lecture                                      |
|--------------------------------------|------------|----------------------------------------------|
| Ancienneté (`tenure`, +1 écart-type) | 0,29       | Facteur le plus protecteur                   |
| Contrat de 2 ans                     | 0,46       | Beaucoup moins de départs                    |
| Internet DSL                         | 0,52       | Moins de départs que la fibre                |
| Internet fibre optique               | 1,89       | Environ 1,9 fois plus de chances de résilier |
| Contrat mensuel                      | 1,79       | Environ 1,8 fois plus de chances de résilier |



**Profil le plus à risque :** contrat mensuel, faible ancienneté, abonnement fibre optique.

**Précautions d'interprétation :**
- Ces résultats décrivent des **associations statistiques, pas des relations de cause à effet**.
- `MonthlyCharges`, `TotalCharges` et `tenure` sont fortement liées entre elles ; leurs coefficients ne doivent pas être lus isolément.
- Les modalités « No internet service » sont une même information répétée par l'encodage.

## 7. Recommandations

1. **Prioriser les clients en contrat mensuel et de faible ancienneté** pour les actions de rétention, notamment pendant les premiers mois d'abonnement.
2. **Encourager le passage à un contrat annuel ou de 2 ans** (offre, avantage tarifaire), l'engagement contractuel étant fortement associé à la fidélité.
3. **Examiner l'offre fibre optique** (prix, qualité de service, concurrence) pour comprendre pourquoi ses clients résilient davantage.
4. **Fixer le seuil du modèle avec les équipes métier**, à partir du coût d'une action de rétention et de la valeur d'un client conservé.


## 8. Structure du dépôt

```
.
├── Telco-Customer-Churn.csv
├── Telco-Customer-Churn.ipynb
├── illustration_churn.png
└── README.md

```

## 9. Exécuter le projet

```bash
pip install numpy pandas scikit-learn seaborn matplotlib imbalanced-learn jupyter
jupyter notebook Telco-Customer-Churn.ipynb
```

---

*Auteure : [Josiane SOBGO DONGO](https://www.linkedin.com/in/josiane-dongo)*