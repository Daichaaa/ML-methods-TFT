import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GroupKFold, cross_val_score

fichier_dataset = 'data/dump.parquet'

print("Chargement du dataset...")
df = pd.read_parquet(fichier_dataset)

y = df['win']
groupes = df['match_id']
X = df.drop(columns=['win', 'match_id', 'placement'], errors='ignore')

# Le Naive Bayes fait la supposition d'indépendance entre les variables, ce qui est rarement le cas en réalité, surtout dans un jeu aussi complexe que TFT. C'est pour ça qu'il est souvent moins performant que des modèles plus sophistiqués comme le Random Forest. Mais c'est toujours intéressant de voir comment il se débrouille.
modele_nb = GaussianNB()

gkf = GroupKFold(n_splits=5)

scores = cross_val_score(modele_nb, X, y, cv=gkf, groups=groupes, scoring='accuracy', n_jobs=-1)

for i, score in enumerate(scores, 1):
    print(f"-> Fold {i} : {score * 100:.2f} %")

print(f"Moyenne absolue : {np.mean(scores) * 100:.2f} %")
print(f"Écart-type : +- {np.std(scores) * 100:.2f} %")

