import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_size = (64, 64)
batch_size = 32
epochs = 20
images_folder = 'images'
augmented_folder = 'augmentation_images'
augmentation_per_image = 5
mode_filename = 'pictogram_cnn_model.keras'

# === DATA AUGMENTATION ===

def augment_data():
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    os.makedirs(augmented_folder, exist_ok=True)

    for file_name in os.listdir(images_folder):
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(images_folder, file_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error reading {img_path}. Skipping...")
            continue

        img = cv2.resize(img, image_size)
        img = img.reshape((image_size[0], image_size[1], 1))

        label = file_name.split('_')[0]
        label_dir = os.path.join(augmented_folder, label)
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


# === CNN MODEL TRAINING ===

def train_model():
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        augmented_folder,
        target_size=image_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        augmented_folder,
        target_size=image_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    model = models.Sequential([
        layers.Input(shape=(64, 64, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(train_generator.num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Training CNN model...")
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs
    )

    model.save(mode_filename)
    print(f"Model saved as '{mode_filename}'")


# === PREDICTION WITH CAMERA ===

def load_class_names():
    return sorted(entry.name for entry in os.scandir(augmented_folder) if entry.is_dir())

def predict_from_frame(model, frame, class_names):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, image_size)
    normalized = resized.astype('float32') / 255.0
    input_image = normalized.reshape(1, 64, 64, 1)

    predictions = model.predict(input_image)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions)

    return class_names[class_index], confidence

def start_camera_prediction():
    if not os.path.exists(mode_filename):
        print("Model not found. Train the model first.")
        return

    print("Loading model...")
    model = tf.keras.models.load_model(mode_filename)
    class_names = load_class_names()
    print(f"Loaded classes: {class_names}")

    # no meu pc: 0 para webcam, 1 para câmara do telemóvel ao usar Camo Studio
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error opening camera.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error capturing frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 50 or h < 50:
                continue

            roi = gray[y:y+h, x:x+w]
            resized = cv2.resize(roi, image_size)
            normalized = resized.astype('float32') / 255.0
            input_image = normalized.reshape(1, 64, 64, 1)

            predictions = model.predict(input_image, verbose=0)
            class_index = np.argmax(predictions)
            confidence = np.max(predictions)

            if confidence > 0.8:
                label = class_names[class_index]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                text = f"{label} ({confidence*100:.1f}%)"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Pictogram Identification - Press Q to quit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# === MAIN MENU ===

if __name__ == '__main__':
    while True:
        print("\nChoose a mode:")
        print("[1] Data augmentation") # este devia sair e o programa fazer automaticamente
        print("[2] Train model")
        print("[3] Identify pictograms with camera")
        print("[0] Quit")
        choice = input("(0/1/2/3): ")

        if choice == '1':
            augment_data()
        elif choice == '2':
            train_model()
        elif choice == '3':
            start_camera_prediction()
        elif choice == '0':
            print("Quitting program...")
            break
        else:
            print("Invalid option. Try again.")