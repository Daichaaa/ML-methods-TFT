from matplotlib import pyplot as plt

fichier_dataset = 'data/dump.parquet'

# On prend les scores données par les fichiers random_forest et naives_bayes
scores_nb = [
    68.11,
    68.93,
    68.45,
    68.58,
    68.52
]

scores_rf = [
    83.84,
    83.89,
    84.02,
    84.27,
    84.27
]

print(f"Scores Random Forest : {scores_rf}")
print(f"Scores Naive Bayes : {scores_nb}")

plt.figure(figsize=(8, 6))

# On regroupe les deux scores dans une liste
donnees_graphique = [scores_rf, scores_nb]

# On crée le boxplot (patch_artist=True permet de colorier l'intérieur)
box = plt.boxplot(donnees_graphique, patch_artist=True,
                  medianprops=dict(color='black', linewidth=2),
                  whiskerprops=dict(color='black'),
                  capprops=dict(color='black'))

# On applique les couleurs par défaut de Matplotlib (C0=Bleu, C1=Orange)
couleurs = ['C0', 'C1']
for patch, couleur in zip(box['boxes'], couleurs):
    patch.set_facecolor(couleur)

plt.title("Performance comparison : Random Forest vs Naive Bayes", fontsize=14, fontweight='bold')
plt.ylabel("Accuracy (%)", fontsize=12)

# On nomme l'axe X pour les deux boîtes
plt.xticks([1, 2], ['Random Forest', 'Naive Bayes'], fontsize=12, fontweight='bold')

# On ajuste l'échelle de l'axe Y pour englober les deux boîtes avec une petite marge
min_global = min(min(scores_rf), min(scores_nb))
max_global = max(max(scores_rf), max(scores_nb))
plt.ylim(min_global - 2, max_global + 2)

# Ajout d'une grille discrète
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('figures/nb_vs_rf.png', dpi=300)