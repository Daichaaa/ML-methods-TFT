import pandas as pd
import plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, \
    precision_recall_fscore_support, RocCurveDisplay

fichier_dataset = 'data/dump.parquet'

print("Chargement du dataset...")
df = pd.read_parquet(fichier_dataset)

y = df['win']

groupes = df['match_id']
X = df.drop(columns=['win', 'match_id', 'placement'])

# 20% de test, 80% de train
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

train_index, test_index = next(gss.split(X, y, groups=groupes))

X_train, X_test = X.iloc[train_index], X.iloc[test_index]
y_train, y_test = y.iloc[train_index], y.iloc[test_index]

print(f"-> Entraînement : {len(X_train)} joueurs")
print(f"-> Test : {len(X_test)} joueurs")

modele_rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=1, max_depth=None, random_state=42, n_jobs=-1)
modele_rf.fit(X_train, y_train)

print("Évaluation des prédictions...")
y_pred = modele_rf.predict(X_test)

precision = accuracy_score(y_test, y_pred)
print(f"Précision globale du modèle : {precision * 100:.2f} %")

cm = confusion_matrix(y_test, y_pred)
print("Matrice de Confusion :\n")
print(cm)

print("Rapport Détaillé :\n")
print(classification_report(y_test, y_pred))

print("Analyse de l'importance des variables (Feature Importance)...")

# Attribut caché du modèle avec l'importance de chaque colonne
importances = modele_rf.feature_importances_

# On crée un petit tableau pour associer le nom de la colonne avec son score
noms_colonnes = X_train.columns
df_importances = pd.DataFrame({
    'Variable': noms_colonnes, 
    'Importance': importances
})

# On trie le tableau pour mettre les variables les plus importantes en haut
df_importances = df_importances.sort_values(by='Importance', ascending=False)

print("Top 20 des variables les plus importantes pour la détermination de la victoire par le modèle :")
for index, row in df_importances.head(20).iterrows():
    # On affiche le nom de la variable, puis son importance transformée en %
    print(f"- {row['Variable']:<30} : {row['Importance']*100:.2f} % d'influence")


# Pyplot figures

# Confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Bot 4 (Defeat)', 'Top 4 (Victory)'])
disp.plot(cmap='Blues')
plt.grid(False)
plt.savefig('figures/confusion_matrix.png', dpi=300, bbox_inches='tight')

# Precision, recall, f1
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
accuracy = modele_rf.score(X_test, y_test)
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [accuracy, precision, recall, f1]

plt.figure(figsize=(8, 5))
bars = plt.bar(metrics_names, metrics_values)
plt.axhline(y=0.5, color='black', linestyle='--', label='Random (50%)')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval*100:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.ylim(0, 1)
plt.title("Global Performances of the Model (Random Forest)")
plt.ylabel("Score")
plt.legend()
plt.savefig('figures/accuracy_precision_recall_f1.png', dpi=300)

# Top 10 most important variables

df_importances = df_importances.sort_values(by='Importance', ascending=False).head(10) # Top 10
plt.figure(figsize=(10, 6))
plt.barh(df_importances['Variable'][::-1], df_importances['Importance'][::-1])
plt.title("Top 10 Most Important Variables in the Random Forest")
plt.xlabel("Importance (Weights)")
plt.tight_layout()
plt.savefig('figures/most_important_variables.png', dpi=300)

# ROC Curve
fig, ax = plt.subplots(figsize=(10, 6))
roc_disp = RocCurveDisplay.from_estimator(
    modele_rf,
    X_test,
    y_test,
    ax= ax,
    name= 'Random Forest',
    curve_kwargs={
        'color': 'darkorange',
        'lw': 2,
    }
)
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC = 0.50)')

ax.set_title("ROC Curve")
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/roc_curve.png', dpi=300)
