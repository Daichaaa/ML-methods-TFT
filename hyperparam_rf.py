import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.ensemble import RandomForestClassifier

fichier_dataset = 'data/dump.parquet'

print("Chargement du dataset...")
df = pd.read_parquet(fichier_dataset)

y = df['win']
groupes = df['match_id']
X = df.drop(columns=['win', 'match_id', 'placement'])

param_grid = {
    'n_estimators': [100, 200],         # Nombre d'arbres dans la forêt
    'max_depth': [None, 15, 30],        # Profondeur max (None = illimité)
    'min_samples_leaf': [1, 5]          # Nombre minimum de joueurs à la fin d'une branche
}

gkf = GroupKFold(n_splits=5)

modele_base = RandomForestClassifier(random_state=42, n_jobs=-1)

grid_search = GridSearchCV(
    estimator=modele_base,
    param_grid=param_grid,
    cv=gkf,                  # On lui donne pour garder les matchs groupés
    scoring='accuracy',
    verbose=2,               # Affiche l'avancement dans la console
    n_jobs=1                 # Laisse le n_jobs=-1 au modèle de base
)

print("Lancement des tests...")
grid_search.fit(X, y, groups=groupes)

print(f"Meilleure précision trouvée : {grid_search.best_score_ * 100:.2f} %")
print("Meilleurs paramètres :")
for param, valeur in grid_search.best_params_.items():
    print(f"   -> {param} : {valeur}")
