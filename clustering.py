import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tft_utils import traduire_synergie, traduire_champion

fichier_dataset = 'data/dump.parquet'

print("Chargement du dataset...")
df = pd.read_parquet(fichier_dataset)

colonnes_synergies = df.filter(like='syn_').columns
colonnes_champions = df.filter(like='unit_').columns

X_cluster = df[colonnes_synergies].fillna(0)

print("Entraînement du K-Means (10 compositions)...")
kmeans = KMeans(n_clusters=10, random_state=42, n_init='auto')
df['cluster'] = kmeans.fit_predict(X_cluster)

print("TOP 10 COMPOSITIONS :\n")
print("-" * 65)

for i in range(10):
    joueurs_cluster = df[df['cluster'] == i]
    winrate = joueurs_cluster['win'].mean() * 100
    
    # On utilise la moyenne juste pour classer l'importance des synergies
    moyennes_syn = joueurs_cluster[colonnes_synergies].mean()
    top_4_syn = moyennes_syn.sort_values(ascending=False).head(4)
    
    # Extraction du Top 5 des Champions
    moyennes_champ = joueurs_cluster[colonnes_champions].mean()
    top_5_champ = moyennes_champ.sort_values(ascending=False).head(5)
    
    print(f"- Composition {i} ({len(joueurs_cluster)} joueurs) - Taux de Top 4 : {winrate:.1f}%")
    
    print("   - Synergies majeures :")
    for trait in top_4_syn.index:
        # On filtre pour ne garder que les joueurs du cluster qui ont cette synergie activée
        joueurs_avec_synergie = joueurs_cluster[joueurs_cluster[trait] > 0]
        
        # S'il y a des joueurs, on cherche le palier le plus fréquent
        if not joueurs_avec_synergie.empty:
            palier_dominant = int(joueurs_avec_synergie[trait].mode().iloc[0])
            pourcentage_joueurs = (len(joueurs_avec_synergie) / len(joueurs_cluster)) * 100
        else:
            palier_dominant = 0
            pourcentage_joueurs = 0

        nom_trait = traduire_synergie(trait.replace('syn_', ''))
        
        # Affichage propre avec le palier réel et le % de joueurs
        print(f"      - {nom_trait:<20} | Palier {palier_dominant} (Joué par {pourcentage_joueurs:.0f}% du groupe)")
            
    print("   - Champions clés :")
    for champ in top_5_champ.index:
        nom_champ = traduire_champion(champ.replace('unit_', '').replace('_tier', ''))
        print(f"      - {nom_champ:<20}")
        
    print("-" * 65)

# Pyplot figures
# Figure 1
noms_manuels = {
    0: "Demacia",
    1: "Slayers",
    2: "Ekko Reroll",
    3: "Shadow Isles",
    4: "Void",
    5: "Yordle",
    6: "Ryze/Ziggs",
    7: "Bilgewater Peeba",
    8: "Lissandra/Voli",
    9: "Yunara/Wukong"
}

winrates_bruts = df.groupby('cluster')['win'].mean() * 100

# On prépare le tableau pour le graphique
data_graphique = pd.DataFrame({
    'ID_Cluster': winrates_bruts.index,
    'Winrate': winrates_bruts.values
})

data_graphique['Compo'] = data_graphique['ID_Cluster'].map(noms_manuels)

data_graphique = data_graphique.sort_values(by='Winrate', ascending=False)

plt.figure(figsize=(10, 6))

couleurs_wr = ["#17C25E" if wr >= 50 else '#e74c3c' for wr in data_graphique['Winrate']]

plt.bar(data_graphique['Compo'], data_graphique['Winrate'], color=couleurs_wr, edgecolor='black', linewidth=0.5)

plt.axhline(y=50, color='black', linestyle='--', linewidth=2, label="Average(50%)")

plt.title("Top 4 percentage by Cluster", fontsize=14, fontweight='bold')
plt.ylabel("Top 4 rate(%)")

min_y = max(0, data_graphique['Winrate'].min() - 5)
max_y = min(100, data_graphique['Winrate'].max() + 5)
plt.ylim(min_y, max_y)

plt.xticks(rotation=30, ha='right', fontsize=11)
plt.legend()
plt.tight_layout()
plt.savefig('figures/clustering_comps.png', dpi=300)


# Figure 2

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster)
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
print("Génération des graphiques pour K=8, 9 et 10...")
valeurs_k = [8, 9, 10]
for i, n_clusters in enumerate(valeurs_k):
    # Entraînement du K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    y_hat = kmeans.fit_predict(X_cluster)

    # Dessin du nuage de points
    ax[i].scatter(X_pca[:, 0], X_pca[:, 1], c=y_hat, cmap='tab10', s=10, alpha=0.7)
    ax[i].set_title(f"K-Means avec K = {n_clusters}", fontweight='bold', fontsize=12)
    ax[i].set_xticks([])  # On retire les graduations
    ax[i].set_yticks([])

plt.suptitle("Evolution of Clusters (K=8 à K=10)",
             fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('figures/clusters_evolution.png', dpi=300)