import csv
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Charger les données d'entraînement/test à partir du fichier CSV
train_data = pd.read_csv('Deep-Learning-models-for-ear-print-detection/data/train_test/ALBP/train_ALBP.csv')
test_data = pd.read_csv('Deep-Learning-models-for-ear-print-detection/data/train_test/ALBP/test_ALBP.csv')
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
joblib.dump(svm, 'svm_ALBP.pkl')


# Confusion Matrix
confusion = confusion_matrix(y_test,y_pred)
fp =confusion.sum(axis =0)- np.diag(confusion)
fn = confusion.sum(axis =1)- np.diag(confusion)
tp =  np.diag(confusion)
tn= confusion.sum()- (fp + fn + tp)

# Calculter le taux d'erreur
frr = fn /(tp + fn)
far = fp / (tn + fp)
eer = (frr + far) / 2
print("Taux de faux rejet (FRR) : {:.2f}%".format(frr.mean() * 100))
print("Taux de fausse acceptation (FAR) : {:.2f}%".format(far.mean() * 100))
print("Taux d'erreur égal (EER) : {:.2f}%".format(eer.mean() * 100))



# Visualisation
import seaborn as sns
import matplotlib.pyplot as plt
# Visualize confusion matrix
plt.figure(figsize=(30, 30))
sns.heatmap(confusion, annot=True, fmt='d', cmap="Blues")
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
