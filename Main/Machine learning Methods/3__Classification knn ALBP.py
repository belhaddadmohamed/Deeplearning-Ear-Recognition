import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib


# Diviser les données d'entraînement/test en variables de caractéristiques (X_train) et variable cible (y_train)
train_data = pd.read_csv('data/train_test/ALBP/train_ALBP.csv')
test_data = pd.read_csv('data/train_test/ALBP/test_ALBP.csv')
print('Train:'+str(len(train_data))+' Test:'+str(len(test_data)))

# Diviser les données d'entraînement en variables de caractéristiques (X_train) et variable cible (y_train)
X_train =   train_data.iloc[:, :-1].values
y_train =   train_data.iloc[:, -1].values

# Créer une instance de modèle KNN avec k=3
knn = KNeighborsClassifier(n_neighbors=1)

# Entraîner le modèle KNN sur les données d'entraînement
print("Training KNN Model...")
knn.fit(X_train, y_train)


X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Prédire les étiquettes des données de test à l'aide du modèle entraîné
y_pred = knn.predict(X_test)

# Save the model
joblib.dump(knn, 'knn_ALBP.pkl')

# # Calculer l'exactitude du modèle KNN sur les données de test
accuracy = accuracy_score(y_test, y_pred)

#
# # Afficher les mesures
print("Exactitude du modèle KNN : {:.2f}%".format(accuracy * 100))
# print("FAR : {:.2f}%".format(far * 100))
# print("FRR : {:.2f}%".format(frr * 100))
# print("EER : {:.2f}%".format(eer * 100))



