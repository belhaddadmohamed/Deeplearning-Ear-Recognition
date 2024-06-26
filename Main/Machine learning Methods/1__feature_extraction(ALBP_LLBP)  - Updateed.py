import os
import cv2
import csv
import math
import numpy as np
import pandas as pd
import random


# ======================================================================================
# Preprocessing Methods

def augmenter_image(image):
    angle = random.randint(-10, 10)
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image


def redimensionner_image(image, width, height):
    return cv2.resize(image, (width, height))


def is_image_file(file_path):
    return file_path.lower().endswith('.png')


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype('int')

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


# =====================================================================================
# Feature Extraction

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

def process_image(image_path, output_folder):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    # Creation of LBP|LLBP|ALBP Folders 
    person_name = os.path.basename(os.path.dirname(image_path))     # Get the person id
    # person_output_folder = os.path.join(output_folder, person_name) # Get the person folder
    person_output_folder = output_folder
    # hog_folder = os.path.join(person_output_folder, 'HOG')
    llbp_folder = os.path.join(person_output_folder, 'LLBP')
    albp_folder = os.path.join(person_output_folder, 'ALBP')
    # os.makedirs(hog_folder, exist_ok=True)
    os.makedirs(llbp_folder, exist_ok=True)
    os.makedirs(albp_folder, exist_ok=True)

    #  (Original Image) Extract and save features ======================================
    # Convert to grayscale and resize
    # Filtering
    filtered_image = cv2.GaussianBlur(image, (5, 5), 0)
    # Gray_Scaling
    gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    # Resizing
    width, height = 100, 150
    resized_image = redimensionner_image(gray_image, width, height)
    # Compute features
    # lbp_image = compute_hog(resized_image)
    # hog = compute_hog(resized_image, cell_size=6, bin_size=5)
    # vector, hog_image = hog.extract()
    llbp_image = compute_llbp(resized_image)
    albp_image = compute_albp(resized_image)
    # Save the features to CSV
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    # hog_image_path = os.path.join(hog_folder, f'hog_features_{base_name}.csv')
    llbp_image_path = os.path.join(llbp_folder, f'llbp_features_{base_name}.csv')
    albp_image_path = os.path.join(albp_folder, f'albp_features_{base_name}.csv')
    # save_image_to_csv(hog_image, hog_image_path)
    save_image_to_csv(llbp_image, llbp_image_path)
    save_image_to_csv(albp_image, albp_image_path)
    # ==================================================================================


    augmented_images = []
    for _ in range(3):  # 2 augmentations par image
        augmented_image = augmenter_image(image)
        augmented_images.append(augmented_image)

    for i, augmented_image in enumerate(augmented_images, start=1):

        # Filtering
        filtered_image = cv2.GaussianBlur(augmented_image, (5, 5), 0)
        # Gray_Scaling
        gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
        # Resizing
        width, height = 100, 150
        resized_image = redimensionner_image(gray_image, width, height)

        # Compute LBP|ALBP|LLBP (Features Extraction)
        # lbp_img = compute_hog(resized_image)
        # hog = compute_hog(resized_image, cell_size=6, bin_size=5)
        # vector, hog_image = hog.extract()
        llbp_img = compute_llbp(resized_image)
        albp_img = compute_albp(resized_image)

        # Save LBP|ALBP|LLBP image to CSV
        # hog_image_path = hog_folder + f'/hog_features_{os.path.splitext(os.path.basename(image_path))[0]}_{i}.csv'
        llbp_image_path = llbp_folder + f'/llbp_features_{os.path.splitext(os.path.basename(image_path))[0]}_{i}.csv'
        albp_image_path = albp_folder + f'/albp_features_{os.path.splitext(os.path.basename(image_path))[0]}_{i}.csv'
        # save_image_to_csv(hog_image, hog_image_path)
        save_image_to_csv(llbp_img, llbp_image_path)
        save_image_to_csv(albp_img, albp_image_path)
        # print(lbp_image_path)
        # print(llbp_image_path)
        # print(albp_image_path)


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