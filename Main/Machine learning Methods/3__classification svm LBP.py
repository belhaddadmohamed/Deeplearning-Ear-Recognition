import csv
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Charger les données d'entraînement/test à partir du fichier CSV
train_data = pd.read_csv('data/train_test/LBP/train_LBP.csv')
test_data = pd.read_csv('data/train_test/LBP/test_LBP.csv')
print('Train:'+str(len(train_data))+' Test:'+str(len(test_data)))


# Diviser les données d'entraînement en variables de caractéristiques (X_train) et variable cible (y_train)
X_train =   train_data.iloc[:, :-1].values
y_train =   train_data.iloc[:, -1].values

y_train_discrete=y_train

# Créer une instance de modèle SVM
svm = SVC(kernel='linear', C=1, gamma='auto')

# Entraîner le modèle SVM sur les données d'entraînement
print("Training SVM Model...")
svm.fit(X_train, y_train_discrete)


# Diviser les données de test en variables de caractéristiques (X_test) et variable cible (y_test)
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Seuiller y_test pour créer une variable cible discrète
y_test_discrete = y_test

# Prédire les étiquettes des données de test à l'aide du modèle entraîné
y_pred = svm.predict(X_test)
# # Calculer l'exactitude du modèle SVM sur les données de test
accuracy = accuracy_score(y_test_discrete, y_pred)

# Afficher l'exactitude du modèle SVM sur les données de test
print("Exactitude du modèle SVM : {:.2f}%".format(accuracy*100))

# Save the model
# joblib.dump(svm, 'svm_model_LBP_me.pkl')
