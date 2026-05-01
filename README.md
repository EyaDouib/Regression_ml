# 🏥 Prédiction des Charges Médicales

> **Machine Learning / Régression** — Projet académique  
> Présentée par : **Dhouib Eya** | Professeur : **Mr. Ben Hamed Bassem**  
> 1 MR ISI — 2025-2026

---

## 📋 Description

Ce projet applique des algorithmes de **régression supervisée** pour prédire les charges médicales individuelles à partir de caractéristiques personnelles (âge, IMC, statut fumeur, etc.). Il couvre l'intégralité du pipeline Data Science : exploration, prétraitement, modélisation, évaluation et sélection de variables.

---

## 📁 Structure du projet

```
📦 regression_ML/
├── regression.ipynb          # Notebook principal (toutes les phases)
├── medical-charges.csv       # Dataset source
└── README.md                 # Ce fichier
```

---

## 📊 Dataset

| Propriété       | Valeur                    |
|-----------------|---------------------------|
| **Fichier**     | `medical-charges.csv`     |
| **Lignes**      | 1 338 (après suppression d'un doublon) |
| **Colonnes**    | 7                         |
| **Cible**       | `charges` (en $)          |

### Variables

| Variable   | Type        | Description                              |
|------------|-------------|------------------------------------------|
| `age`      | Numérique   | Âge de l'individu                        |
| `sex`      | Catégorielle| Genre (male / female)                    |
| `bmi`      | Numérique   | Indice de Masse Corporelle               |
| `children` | Numérique   | Nombre d'enfants à charge                |
| `smoker`   | Catégorielle| Statut fumeur (yes / no)                 |
| `region`   | Catégorielle| Région géographique (4 valeurs)          |
| `charges`  | Numérique   | **Variable cible** — charges médicales ($)|

---

## 🔬 Pipeline du projet

### Phase 1 — Analyse Exploratoire (EDA)

- **Analyse univariée** : histogrammes et boxplots des variables numériques (`age`, `bmi`, `children`, `charges`) ; countplots des variables catégorielles (`sex`, `smoker`, `region`).
- **Analyse bivariée** : scatter plots colorés par statut fumeur, boxplots des charges par variable catégorielle, interaction `bmi × smoker`.
- **Analyse multivariée** : matrice de corrélation.
- **Détection des outliers** via IQR.

**Observations clés :**
- Corrélation très forte entre `smoker` et `charges` → **0.79**
- Corrélation modérée : `age` (0.30), `bmi` (0.20)
- Impact très faible : `children` et `sex`
- Les fumeurs paient en moyenne **~3.8× plus** de charges que les non-fumeurs

---

### Phase 2 — Nettoyage et Prétraitement

| Étape                        | Action                                                        |
|------------------------------|---------------------------------------------------------------|
| **Valeurs manquantes**        | Aucune détectée                                               |
| **Doublons**                  | 1 doublon supprimé → dataset final : (1 337, 7)               |
| **Encodage catégoriel**       | `LabelEncoder` pour `sex`, `smoker` ; One-Hot pour `region`   |
| **Feature Scaling**           | `StandardScaler` sur `age`, `bmi`, `children`                 |
| **Train / Test Split**        | 80% / 20% — `random_state=42` — X_train:(1069,8), X_test:(268,8) |

---

### Phase 3 — Modélisation

Quatre modèles de régression ont été entraînés et comparés :

| Modèle               | Type              | Avantage principal                          |
|----------------------|-------------------|---------------------------------------------|
| **Linear Regression**| OLS classique     | Simple et interprétable                     |
| **Ridge (L2)**       | Régularisation L2 | Meilleure stabilité, évite le surapprentissage |
| **Lasso (L1)**       | Régularisation L1 | Sélection automatique de variables          |
| **ElasticNet (L1+L2)**| Combinaison      | Flexible pour de nombreuses variables       |

Les hyperparamètres `alpha` ont été optimisés par **validation croisée** (`RidgeCV`, `LassoCV`, `ElasticNetCV`).

---

### Phase 4 — Évaluation des Modèles

#### Métriques utilisées

| Métrique    | Description                                             | Objectif        |
|-------------|---------------------------------------------------------|-----------------|
| **RMSE**    | Erreur quadratique moyenne (en $) — punit les grosses erreurs | Minimiser |
| **MAE**     | Erreur absolue moyenne (en $) — plus interprétable      | Minimiser       |
| **R²**      | Proportion de variance expliquée (0 à 1)                | Maximiser       |
| **CV R²**   | R² moyen sur 5 folds — mesure la stabilité              | Maximiser       |
| **Adj. R²** | R² ajusté au nombre de variables                        | Maximiser       |

#### Résultats

| Modèle               | RMSE Test  | MAE Test   | R² Test | CV R² (5-fold)      | Meilleur α        |
|----------------------|------------|------------|---------|---------------------|-------------------|
| **Linear Regression**| **5 956**  | **4 177**  | **0.8069** | 0.7258 ± 0.0253 | —                 |
| Ridge Regression     | 5 987      | 4 210      | 0.8049  | 0.7259 ± 0.0252     | 1.98              |
| Lasso Regression     | 5 993      | 4 198      | 0.8045  | 0.7260 ± 0.0251     | 26.43             |
| ElasticNet           | 5 993      | 4 198      | 0.8045  | 0.7260 ± 0.0251     | 26.43 (l1_ratio=1)|

> ✅ **Meilleur modèle : Régression Linéaire (OLS)** avec R² = 0.8069.  
> La régularisation n'apporte pas d'amélioration significative, ce qui s'explique par la taille modeste du dataset et l'absence de forte multicolinéarité.

---

### Phase 5 — Sélection de Features

Trois méthodes de sélection ont été appliquées :

| Méthode                      | Features retenues                          | Features supprimées          |
|------------------------------|--------------------------------------------|------------------------------|
| **Backward Elimination** (p-values) | `age`, `bmi`, `children`, `smoker`   | `sex` + 3 régions            |
| **RFE** (Recursive Feature Elimination) | `age`, `bmi`, `children`, `smoker` | `sex` + 3 régions            |
| **Lasso** (coefficients = 0) | `age`, `bmi`, `children`, `smoker` + 2 régions | `sex`, `region_northwest` |

> **Conclusion** : les 4 variables les plus importantes sont `age`, `bmi`, `children` et `smoker`.

---

### Phase 6 — Comparaison All Features vs Selected Features

Les modèles ont été ré-entraînés sur les 4 features sélectionnées. La comparaison montre une performance très similaire avec un modèle plus simple et plus interprétable.

---

## 📌 Observations clés

| # | Observation |
|---|-------------|
| 1 | **`smoker`** est la variable dominante — corrélation 0.79, coefficient ~23 000$ dans tous les modèles |
| 2 | **`age`** et **`bmi`** ont un impact positif significatif sur les charges |
| 3 | **`sex`** et **`region`** apportent très peu d'information prédictive |
| 4 | Les quatre modèles de régression donnent des performances très proches (R² ≈ 0.80) |
| 5 | La **Régression Linéaire simple** est le modèle le plus performant dans ce contexte |

---

## ⚠️ Limites identifiées

| Limite                     | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| **Taille du dataset**       | 1 337 observations limitent la généralisation sur une large population      |
| **Hypothèse de linéarité** | L'interaction `bmi × smoker` est non-linéaire — mal capturée par OLS        |
| **Variables manquantes**    | Antécédents médicaux, revenu, mode de vie absents du dataset                |
| **Hétéroscédasticité**      | La variance des résidus croît avec les charges élevées — hypothèse OLS violée |
| **Pas de transformation**   | `log(charges)` n'a pas été testée — pourrait améliorer les performances     |

---

## 🚀 Perspectives d'amélioration

| Piste                         | Description                                                              |
|-------------------------------|--------------------------------------------------------------------------|
| `log(charges)`                | Réduire l'hétéroscédasticité et améliorer la normalité des résidus       |
| Feature Engineering           | Créer une interaction explicite `bmi × smoker`                           |
| Modèles non-linéaires         | Random Forest, XGBoost — mieux adaptés aux relations complexes           |
| Données supplémentaires        | Antécédents médicaux, historique de remboursements, mode de vie          |

---

## 🛠️ Technologies utilisées

```python
pandas          # Manipulation des données
numpy           # Calcul numérique
matplotlib      # Visualisation
seaborn         # Visualisation statistique
scikit-learn    # Modèles ML, preprocessing, évaluation, sélection de features
statsmodels     # Régression OLS, backward elimination (p-values)
scipy           # Tests statistiques (Shapiro, Q-Q plots)
```

---

## ▶️ Exécution

```bash
# 1. Cloner ou télécharger le projet
# 2. Installer les dépendances
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy

# 3. Placer le fichier medical-charges.csv dans le même répertoire que le notebook

# 4. Lancer Jupyter
jupyter notebook regression.ipynb
```

---

*Projet réalisé dans le cadre du cours de Machine Learning — 1 MR ISI, 2025-2026.*
