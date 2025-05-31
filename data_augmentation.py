import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# isto foi para testar a parte de data augmentation sozinha
images_folder = 'images'
augmentation_folder = 'augmentation_images'
image_size = (64, 64)
augmentation_per_image = 5

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

os.makedirs(augmentation_folder, exist_ok=True)

for file_name in os.listdir(images_folder):
    if not (file_name.endswith('.png') or file_name.endswith('.jpg') or file_name.endswith('.jpeg')):
        continue

    img_path = os.path.join(images_folder, file_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Erro ao ler {img_path}, pulando")
        continue

    img = cv2.resize(img, image_size)
    img = img.reshape((image_size[0], image_size[1], 1))

    label = file_name.split('_')[0]
    label_dir = os.path.join(augmentation_folder, label)
    os.makedirs(label_dir, exist_ok=True)

    original_save_path = os.path.join(label_dir, file_name)
    cv2.imwrite(original_save_path, img)

    i = 0
    for batch in datagen.flow(np.array([img]), batch_size=1):
        augmented_img = batch[0].reshape(image_size)
        new_name = f"{label}_{i:03d}.png"
        new_path = os.path.join(label_dir, new_name)
        cv2.imwrite(new_path, augmented_img)

        i += 1
        if i >= augmentation_per_image:
            break

print("Data augmentation concluded!")