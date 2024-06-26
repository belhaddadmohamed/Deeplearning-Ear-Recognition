import os
import cv2
import numpy as np
import random

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

def afficher_image(image, title='Image'):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_image(image_path, augmentation_folder, filtrage_folder, niveaux_de_gris_folder, redimensionnement_folder):
    # Charger l'image
    image = cv2.imread(image_path)

    # ============================(1)Preprocess_original_img==============================
    filtered_image = cv2.GaussianBlur(image, (5, 5), 0)
    gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    resized_image = redimensionner_image(gray_image, 100, 150)
    output_path = os.path.join(redimensionnement_folder, f'resized_{os.path.basename(image_path)}')
    cv2.imwrite(output_path, resized_image)
    # ===================================================================================

    augmented_images = []
    for _ in range(2):  # 2 augmentations par image
        # Appliquer l'augmentation
        augmented_image = augmenter_image(image)
        augmented_images.append(augmented_image)

    # Enregistrer et afficher les images augmentées
    for i, augmented_image in enumerate(augmented_images, start=1):
        # Sauvegarder dans le dossier d'augmentation
        output_path = os.path.join(augmentation_folder, f'augmented_{os.path.basename(image_path)}_{i}.png')
        cv2.imwrite(output_path, augmented_image)

        # Appliquer le filtrage
        filtered_image = cv2.GaussianBlur(augmented_image, (5, 5), 0)
        output_path = os.path.join(filtrage_folder, f'filtered_{os.path.basename(image_path)}_{i}.png')
        cv2.imwrite(output_path, filtered_image)

        # Convertir en niveaux de gris
        gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
        output_path = os.path.join(niveaux_de_gris_folder, f'gray_{os.path.basename(image_path)}_{i}.png')
        cv2.imwrite(output_path, gray_image)

        # Redimensionner l'image
        width, height =100, 150 # Nouvelles dimensions
        resized_image = redimensionner_image(gray_image, width, height)
        output_path = os.path.join(redimensionnement_folder, f'resized_{os.path.basename(image_path)}_{i}.png')
        cv2.imwrite(output_path, resized_image)

def main():
    database_folder = 'AMI'
    output_folder = 'affichage pretraitement'

    os.makedirs(output_folder, exist_ok=True)

    # Parcourir les dossiers de la base de données AMI
    for person_folder in os.listdir(database_folder):
        person_folder_path = os.path.join(database_folder, person_folder)
        print(person_folder_path)
        if os.path.isdir(person_folder_path):
            # Créer le dossier de sortie pour la personne spécifique
            person_output_folder = os.path.join(output_folder, person_folder)
            os.makedirs(person_output_folder, exist_ok=True)

            # Créer les sous-dossiers pour chaque prétraitement
            augmentation_folder = os.path.join(person_output_folder, 'augmentation')
            filtrage_folder = os.path.join(person_output_folder, 'filtrage')
            niveaux_de_gris_folder = os.path.join(person_output_folder, 'niveaux_de_gris')
            redimensionnement_folder = os.path.join(person_output_folder, 'redimensionnement')
            os.makedirs(augmentation_folder, exist_ok=True)
            os.makedirs(filtrage_folder, exist_ok=True)
            os.makedirs(niveaux_de_gris_folder, exist_ok=True)
            os.makedirs(redimensionnement_folder, exist_ok=True)

            # Parcourir les images dans le dossier de la personne spécifique
            for image_file in os.listdir(person_folder_path):
                if is_image_file(image_file):
                    image_path = os.path.join(person_folder_path, image_file)
                    process_image(image_path, augmentation_folder, filtrage_folder, niveaux_de_gris_folder, redimensionnement_folder)   # Preprocess images and place them in their folders

if __name__ == "__main__":
    main()