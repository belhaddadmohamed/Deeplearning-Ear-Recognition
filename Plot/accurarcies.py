import matplotlib.pyplot as plt

# Données
methods = ['KNN ALBP', 'KNN LLBP', 'SVM ALBP', 'SVM LLBP', 'CNN']
accuracies = [85, 90, 95.50, 98.50, 100]
colors = ['skyblue', 'lightgreen', 'orange', 'lightcoral', 'lightpink']

# Plot
plt.figure(figsize=(8, 6))
bars = plt.bar(methods, accuracies, color=colors)
plt.title('Précision par méthode et extracteur de caractéristiques')
plt.xlabel('Méthode - Extracteur de caractéristiques - CRR')
plt.ylabel('Précision (%)')

# Ajouter des annotations
for bar, accuracy in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{accuracy}%', ha='center')

plt.ylim(0, 105)  # Limiter l'axe y entre 0 et 105 pour une meilleure visualisation
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
