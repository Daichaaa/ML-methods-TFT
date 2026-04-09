import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from tft_utils import traduire_item, api_champion

fichier_dataset = 'data/dump.parquet'

print("Chargement du dataset...")
df = pd.read_parquet(fichier_dataset)

y = df['placement']
X = df.drop(columns=['placement', 'win', 'match_id'])

# On entraîne le modèle sur tout le jeu de données 
modele_lr = LinearRegression()
modele_lr.fit(X, y)

# On stocke tous les coefficients (Deltas) dans un tableau
df_coeff = pd.DataFrame({
    'Variable': X.columns,
    'Delta': modele_lr.coef_
})

# Variable cible pour laquelle on veut voir les meilleurs et pires items, ici "Ziggs" en exemple,
CHAMPION_CIBLE = "Ziggs"
CHAMPION_CIBLE_API = api_champion(CHAMPION_CIBLE)

print(f"\nRecherche des Deltas d'items pour : {CHAMPION_CIBLE}")

# On filtre les colonnes qui contiennent "NomDuChampion_with_"
df_champion = df_coeff[df_coeff['Variable'].str.contains(f"{CHAMPION_CIBLE_API}_with_", na=False)]

if df_champion.empty:
    print(f"Aucun combo trouvé pour {CHAMPION_CIBLE} dans les données.")
else:
    # On trie du meilleur (delta négatif) au pire (delta positif)
    df_champion = df_champion.sort_values(by='Delta')
    df_champion['ItemName'] = df_champion['Variable'].apply(lambda variable: traduire_item(variable.replace(f"{CHAMPION_CIBLE_API}_with_", "")))

    print(f"Meilleurs items sur {CHAMPION_CIBLE}")

    # On affiche les items avec un delta négatif (qui font gagner des places)
    items_positifs = df_champion[df_champion['Delta'] < 0]
    for index, row in items_positifs.iterrows():
        nom_propre = row['ItemName']
        print(f"{nom_propre:<25} : {row['Delta']:.3f} places")

    print("\n")

    print(f"Pire items sur {CHAMPION_CIBLE}")

    # On affiche les items avec un delta positif (qui font perdre des places)
    items_negatifs = df_champion[df_champion['Delta'] >= 0].sort_values(by='Delta', ascending=False)
    for index, row in items_negatifs.iterrows():
        nom_propre = row['ItemName']
        print(f"{nom_propre:<25} : +{row['Delta']:.3f} places")

    # Pyplot figure
    plt.figure(figsize=(10, 6))
    colors = ['C2' if val < 0 else 'C3' for val in df_champion['Delta'][::-1]]

    plt.barh(df_champion['ItemName'][::-1], df_champion['Delta'][::-1], color=colors)
    plt.axvline(x=0, color='black', linewidth=1)

    plt.title(f"Deltas of Items for {CHAMPION_CIBLE}", fontweight='bold')
    plt.xlabel("Impact on Placement (Negative Delta = Better)")
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('figures/bis_item_ziggs.png', dpi=300)
