import pandas as pd
import json

from matplotlib import pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
import pyarrow as pa # make sure this package is installed for the .parquet save

input_file = 'data/dump.json'
output_file = 'data/dump.parquet'

with open(input_file, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# Création du dataframe à partir des données du fichier JSON
rows = []
for match in raw_data:
    # Vérification de la structure du fichier
    if 'metadata' not in match or 'info' not in match:
        print("Fichier .json invalide !")
        continue

    # Pour chaque joueur du match, on crée une ligne dans le dataframe
    for participant in match['info']['participants']:
        # Extraction de la target
        row = {
            'win': int(participant['win']),
            'match_id': match['metadata']['match_id'],
            'game_length': match['info']['game_length'],
            'placement': participant['placement'],
            'level': participant['level']
        }

        units = participant.get('units', [])

        # Valeur estimée du plateau
        # Calcul de la valeur en PO (un champion 2* coûte 3 fois le prix du 1*)
        valeur_plateau_estimee = 0
        for unit in units:
            cout_base = unit.get('rarity', 0) + 1
            valeur_unite = cout_base * (3 ** (unit['tier'] - 1))
            valeur_plateau_estimee += valeur_unite

        row['valeur_plateau'] = valeur_plateau_estimee

        # Extraction des synergies
        for trait in participant.get('traits', []):
            if trait['tier_current'] > 0: # On ne garde que les synergies qui ont au moins 1 palier d'activé
                row[f"syn_{trait['name']}"] = trait['tier_current']

        # Extraction des champions et des items
        # On extrait le niveau d'étoiles du champion et le nombre d'items équipé
        combinaisons_joueur = []
        for unit in participant.get('units', []):
            # Champion tier row
            unit_name = unit['character_id']
            tier = unit['tier']
            row[f'unit_{unit_name}_tier'] = max(tier, row.get(f'unit_{unit_name}_tier', tier))

            # Items
            for item_name in unit.get('itemNames', []):
                if "Emblem" not in item_name:
                    # On pourrait faire directement la méthode suivante, mais ça risque de créer trop de 0 et prendre trop de RAM
                    # row[f'{unit_name}_with_{item_name}'] = 1
                    # On va utiliser une matrice creuse "sparse" de pandas plus tard
                    # Pour l'instant on va simplement créer une colonne avec toutes les combinaisons
                    combinaisons_joueur.append(f'{unit_name}_with_{item_name}')

        row['combinaisons'] = combinaisons_joueur # la fameuse liste de combinaisons unités/items qu'on développera plus tard
        rows.append(row)

df = pd.DataFrame(rows)
df.fillna(0, inplace=True)

# Valeurs relatives aux autres joueurs. On insère manuellement dans le tableau pour avoir ces données dans les premières colonnes
moyenne_plateau = df.groupby('match_id')['valeur_plateau'].transform('mean')
valeurs_ratio_plateau = df['valeur_plateau'] / moyenne_plateau
idx_plateau = df.columns.get_loc('valeur_plateau') + 1
df.insert(idx_plateau, 'ratio_valeur_plateau', valeurs_ratio_plateau)

moyenne_level = df.groupby('match_id')['level'].transform('mean')
valeurs_ratio_level = df['level'] / moyenne_level
idx_level = df.columns.get_loc('level') + 1
df.insert(idx_level, 'ratio_level', valeurs_ratio_level)

# On utilise le MLB pour développer la colonne combinaisons en une colonne par combinaison,
# sans utiliser trop de RAM - les 0 ne sont pas en mémoire (il retient "que des 0" à part ces x lignes)
print("Transformation des combinaisons d'objets (MultiLabelBinarizer)...")
mlb = MultiLabelBinarizer(sparse_output=True)
matrice_sparse = mlb.fit_transform(df['combinaisons'])

df_items = pd.DataFrame.sparse.from_spmatrix(
    matrice_sparse,
    index=df.index,
    columns=mlb.classes_
)

df.drop(columns=['combinaisons'], inplace=True)
df = pd.concat([df, df_items], axis=1)

# Pyplot figure - On va montrer le filtrage ci-dessous
plt.figure(figsize=(10, 5))
plt.hist(df['valeur_plateau'], bins=100, range=(0, 300), edgecolor='black', alpha=0.7)
plt.axvline(x=28, color='red', linestyle='--', linewidth=2, label='Threshold for Data Removal for Top 8')
plt.axvline(x=30, color='gray', linestyle='--', linewidth=2, label='Threshold for Data Removal for Top 7')
plt.axvline(x=40, color='gray', linestyle='--', linewidth=2, label='Threshold for Data Removal for Top 4,5,6')
plt.axvline(x=50, color='gray', linestyle='--', linewidth=2, label='Threshold for Data Removal for Top 3')
plt.axvline(x=60, color='red', linestyle='--', linewidth=2, label='Threshold for Data Removal for Top 1,2')
plt.title("Distribution of Total Board Value (before cleaning rage-sell boards)")
plt.xlabel("Total Board Value (Gold)")
plt.ylabel("Number of Matches")
plt.legend()
plt.savefig('figures/distribution_total_board_value.png', dpi=300)

# Filtrage rage-sell
print("Nettoyage des plateaux vendus (rage-sell)...")
nb_lignes_avant = len(df)
seuils_minimums = {1: 60, 2: 60, 3: 50, 4: 40, 5: 40, 6: 40, 7: 30, 8: 28}
seuils_pour_chaque_joueur = df['placement'].astype(int).map(seuils_minimums)
df = df[df['valeur_plateau'] >= seuils_pour_chaque_joueur]
df = df.reset_index(drop=True)
print(f"-> {nb_lignes_avant - len(df)} parties suspectes supprimées !")

# Filtrage des combinaisons ultra-rare
print("Filtrage des combinaisons rares (Feature Selection)...")
cols_combinaisons = [c for c in df.columns if "_with_" in c] # Uniquement sur les combinaisons d'items
seuil_minimum = len(df) * 0.003
frequence_combos = df[cols_combinaisons].sum()
cols_a_supprimer = frequence_combos[frequence_combos < seuil_minimum].index.tolist()
df.drop(columns=cols_a_supprimer, inplace=True)
print(f"-> {len(cols_a_supprimer)} combinaisons trop rares supprimées.")
print(f"-> {len(cols_combinaisons) - len(cols_a_supprimer)} combinaisons conservées.")

# Réduction de la RAM et du stockage utilisés, y comprit lors des futures étapes (ml)
cols_binaires = [c for c in df.columns if "_with_" in c or "win" in c]
cols_petits_entiers = [c for c in df.columns if "tier" in c or "placement" in c or "level" in c]

df[cols_binaires] = df[cols_binaires].astype('int8')
df[cols_petits_entiers] = df[cols_petits_entiers].astype('int8')
df['valeur_plateau'] = df['valeur_plateau'].astype('int32')

print(f"Taille finale en RAM : {df.memory_usage().sum() / 1024**2:.2f} MB")
df.to_parquet(output_file, index=False, engine='pyarrow')