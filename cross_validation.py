import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

fichier_dataset = 'data/dump.parquet'

print("Chargement du dataset...")
df = pd.read_parquet(fichier_dataset)

y = df['win']
groupes = df['match_id']
X = df.drop(columns=['win', 'match_id', 'placement'])

print("Configuration du GroupKFold (5 Folds)...")
# On utilise GroupKFold pour garder les lobbys intacts
gkf = GroupKFold(n_splits=5)

# On garde le modèle Random Forest optimisé avec les hyperparamètres
modele_rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=1, max_depth=None, random_state=42, n_jobs=-1)

print("Entraînement et évaluation des 5 modèles en cours...")
scores = cross_val_score(modele_rf, X, y, cv=gkf, groups=groupes, scoring='accuracy', n_jobs=-1)

print("Résultats :\n")

# On affiche le score de chaque "morceau" pour voir si le modèle est stable
for i, score in enumerate(scores, 1):
    print(f"-> Fold {i} : {score * 100:.2f} %")

print("-" * 40)
print(f"Précision moyenne absolue : {np.mean(scores) * 100:.2f} %")
print(f"Marge d'erreur (Écart-type) : +- {np.std(scores) * 100:.2f} %")