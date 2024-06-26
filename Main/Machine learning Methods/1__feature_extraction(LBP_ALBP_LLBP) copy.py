import os
import cv2
import csv
import numpy as np
import pandas as pd
import random
# ============================
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import numpy as np



# =====================================================================================
# CSV_Extraction Method

def save_image_to_csv(image, file_path):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(image)

def load_image_from_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
        image = np.array(data).astype(np.uint8)
    return image


# ======================================================================================
# Function to read an image from file
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return image

def is_image_file(file_path):
    return file_path.lower().endswith('.png')


# =====================================================================================
# Feature Extraction Methods

def compute_lbp(image):
    # Définir les valeurs de LBP aléatoires pour chaque direction
    random_lbps = np.random.randint(0, 256, (8, 3, 3))

    # Étendre les dimensions de l'image pour faciliter le calcul des LBP
    extended_image = np.pad(image, ((1, 1), (1, 1)), mode='constant')

    # Calculer les valeurs LBP pour chaque pixel de l'image
    lbp_img = np.zeros_like(image)
    for i in range(1, image.shape[0] + 1):
        for j in range(1, image.shape[1] + 1):
            grid = extended_image[i - 1:i + 2, j - 1:j + 2]
            lbp_value = 0
            for k in range(8):
                mask = random_lbps[k] >= grid
                lbp_value += 2**k * np.sum(mask)
            lbp_img[i - 1, j - 1] = lbp_value

    return lbp_img

def compute_llbp(image):
    llbp_image = np.zeros_like(image, dtype=np.float32)
    weights = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            value = 0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    value += image[i + m, j + n] * weights[m + 1, n + 1]
            llbp_image[i, j] = np.clip(value, 0, 255)
    return llbp_image

def compute_albp(image):
    albp_image = np.zeros_like(image)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            center = image[i, j]
            value = 0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    if m != 0 or n != 0:
                        value += int(image[i + m, j + n] >= center) * 2**((3 * (m + 1)) + (n + 1))
            albp_image[i, j] = np.clip(value, 0, 255)
    return albp_image



# =====================================================================================
# Application

# Define the augmentation configuration
datagen = ImageDataGenerator(
    rotation_range=30,  # Rotate up to 30 degrees
    width_shift_range=0.2,  # Shift width by up to 20%
    height_shift_range=0.2,  # Shift height by up to 20%
    shear_range=0.2,  # Shear transformation
    zoom_range=0.2,  # Zooming transformation
    horizontal_flip=True,  # Horizontal flip
    fill_mode='nearest'  # Fill empty pixels after transformation
)

# Function to load and augment an image
def augment_image(image, num_variations=4):
    augmented_images = []
    image_expanded = np.expand_dims(image, axis=0)  # Add batch dimension
    
    # Generate 'num_variations' of augmented images
    augmentation_generator = datagen.flow(image_expanded, batch_size=1)
    for _ in range(num_variations):
        augmented_images.append(augmentation_generator[0][0])  # Get the augmented image
        
    return augmented_images

# Function to resize an image
def resize_image(image, width, height):
    return cv2.resize(image, (width, height))

# Save image data to CSV
# def save_image_to_csv(image, file_path):
#     np.savetxt(file_path, image, delimiter=',')

# Main function to process and augment an image
def process_image(image_path, output_folder):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Create necessary output directories
    lbp_folder = os.path.join(output_folder, 'LBP')
    llbp_folder = os.path.join(output_folder, 'LLBP')
    albp_folder = os.path.join(output_folder, 'ALBP')
    os.makedirs(lbp_folder, exist_ok=True)
    os.makedirs(llbp_folder, exist_ok=True)
    os.makedirs(albp_folder, exist_ok=True)

    #  (Original Image) Extract and save features ======================================
    # Convert to grayscale and resize
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = resize_image(gray_image, 100, 150)
    # Compute features
    lbp_image = compute_lbp(resized_image)
    llbp_image = compute_llbp(resized_image)
    albp_image = compute_albp(resized_image)
    # Save the features to CSV
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    lbp_image_path = os.path.join(lbp_folder, f'lbp_features_{base_name}.csv')
    llbp_image_path = os.path.join(llbp_folder, f'llbp_features_{base_name}.csv')
    albp_image_path = os.path.join(albp_folder, f'albp_features_{base_name}.csv')
    save_image_to_csv(lbp_image, lbp_image_path)
    save_image_to_csv(llbp_image, llbp_image_path)
    save_image_to_csv(albp_image, albp_image_path)
    # Save features to PNG
    # base_name = os.path.splitext(os.path.basename(image_path))[0]
    # lbp_image_path = os.path.join(lbp_folder, f'lbp_features_{base_name}.png')
    # llbp_image_path = os.path.join(llbp_folder, f'llbp_features_{base_name}.png')
    # albp_image_path = os.path.join(albp_folder, f'albp_features_{base_name}.png')
    # cv2.imwrite(lbp_image_path, lbp_image)
    # cv2.imwrite(llbp_image_path, llbp_image)
    # cv2.imwrite(albp_image_path, albp_image)
    # ==================================================================================

    # Augment the image
    augmented_images = augment_image(image, num_variations=4)

    # Process each augmented image
    for i, augmented_image in enumerate(augmented_images, start=1):
        # Convert to grayscale and resize
        gray_image = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2GRAY)
        resized_image = resize_image(gray_image, 100, 150)

        # Compute features
        lbp_image = compute_lbp(resized_image)
        llbp_image = compute_llbp(resized_image)
        albp_image = compute_albp(resized_image)

        # Save the features to CSV
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        lbp_image_path = os.path.join(lbp_folder, f'lbp_features_{base_name}_{i}.csv')
        llbp_image_path = os.path.join(llbp_folder, f'llbp_features_{base_name}_{i}.csv')
        albp_image_path = os.path.join(albp_folder, f'albp_features_{base_name}_{i}.csv')
        save_image_to_csv(lbp_image, lbp_image_path)
        save_image_to_csv(llbp_image, llbp_image_path)
        save_image_to_csv(albp_image, albp_image_path)

        # Save features to PNG
        # base_name = os.path.splitext(os.path.basename(image_path))[0]
        # lbp_image_path = os.path.join(lbp_folder, f'lbp_features_{base_name}_{i}.png')
        # llbp_image_path = os.path.join(llbp_folder, f'llbp_features_{base_name}_{i}.png')
        # albp_image_path = os.path.join(albp_folder, f'albp_features_{base_name}_{i}.png')
        # cv2.imwrite(lbp_image_path, lbp_image)
        # cv2.imwrite(llbp_image_path, llbp_image)
        # cv2.imwrite(albp_image_path, albp_image)



def preprocess_images(database_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for person_folder in os.listdir(database_folder):
        person_folder_path = os.path.join(database_folder, person_folder)   # data/AMI/X
        print(person_folder_path)
        if os.path.isdir(person_folder_path):
            person_output_folder = os.path.join(output_folder, person_folder)   # output_folder/X
            os.makedirs(person_output_folder, exist_ok=True)

            for image_file in os.listdir(person_folder_path):
                if is_image_file(image_file):
                    image_path = os.path.join(person_folder_path, image_file)   # data/AMI/X/img.png
                    process_image(image_path, person_output_folder)


def main():
    database_folder = 'C:/Users/PC-MOH/Desktop/Rahmani Deep leanring/AMI'
    preprocessed_folder = 'data/extract_features'
    # lbp_folder = 'data/lbp_features'

    preprocess_images(database_folder, preprocessed_folder)
    # preprocess_images(preprocessed_folder, lbp_folder)


if __name__ == "__main__":
    main()