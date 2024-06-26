import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set parameters
img_height = 150
img_width = 100
num_classes = 100  # Assuming 100 classes in your dataset

# Set up data augmentation
data_augmentation = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    rescale=1.0 / 255.0
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0).flow_from_directory(
    'CNN_Training_img/train_test_img_1900x300/test_preprocess',
    target_size=(img_height, img_width),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False  # Ensure the data is not shuffled to match predictions with true labels
)

# Load the model
model = tf.keras.models.load_model("CNN_Training_img/my_cnn_model_img150x100.h5")

# Predict on test data
predictions = model.predict(test_datagen)
y_true = test_datagen.classes
y_pred = np.argmax(predictions, axis=1)

# Confusion Matrix
confusion = confusion_matrix(y_true, y_pred)

# Calculate FP, FN, TP, TN
fp = confusion.sum(axis=0) - np.diag(confusion)  
fn = confusion.sum(axis=1) - np.diag(confusion)
tp = np.diag(confusion)
tn = confusion.sum() - (fp + fn + tp)

# Calculate FAR, FRR, EER
far = fp / (fp + tn)
frr = fn / (fn + tp)
eer = (far + frr) / 2

print("False Acceptance Rate (FAR): {:.2f}%".format(far.mean() * 100))
print("False Rejection Rate (FRR): {:.2f}%".format(frr.mean() * 100))
print("Equal Error Rate (EER): {:.2f}%".format(eer.mean() * 100))

# Visualize confusion matrix
plt.figure(figsize=(30, 30))
sns.heatmap(confusion, annot=True, fmt='d', cmap="Blues", 
            xticklabels=test_datagen.class_indices.keys(), 
            yticklabels=test_datagen.class_indices.keys())
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
