import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_size = (64, 64)
batch_size = 32
epochs = 40
images_folder = 'images'
augmented_folder = 'augmentation_images'
augmentation_per_image = 30
mode_filename = 'pictogram_cnn_model.keras'


# === DATA AUGMENTATION ===

def augment_data():
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        zoom_range=0.2,
        brightness_range=[0.7, 1.3],
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

        label = os.path.splitext(file_name)[0].split('_')[0]
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
    datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

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

    if os.path.exists(mode_filename):
        print("Loading existing model for continued training...")
        model = tf.keras.models.load_model(mode_filename)
    else:
        print("Creating new model...")
        model = models.Sequential([
            layers.Input(shape=(64, 64, 1)),

            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dense(256, activation='relu'),
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


def extract_pictogram_roi(frame):
    """
    Extrai apenas o ROI do pictograma mais proeminente na imagem.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Threshold adaptativo para separar pictograma do fundo
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7
    )
    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, frame
    # Selecionar o maior contorno (assumindo que é o pictograma)
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    # Ignorar contornos pequenos
    if w < 30 or h < 30 or area < 1000:
        return None, frame
    # Recorte do ROI
    roi = gray[y:y + h, x:x + w]
    # Ajustar para quadrado
    delta = abs(h - w)
    top, bottom, left, right = 0, 0, 0, 0
    if h > w:
        left = delta // 2
        right = delta - left
    else:
        top = delta // 2
        bottom = delta - top
    roi_square = cv2.copyMakeBorder(roi, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)
    # Desenhar retângulo no frame original para feedback visual
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    return roi_square, frame


def start_camera_prediction():
    if not os.path.exists(mode_filename):
        print("Model not found. Train the model first.")
        return

    print("Loading model...")
    model = tf.keras.models.load_model(mode_filename)
    class_names = load_class_names()
    print(f"Loaded classes: {class_names}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening camera.")
        return

    print("Press 's' to start forming the sentence.")
    print("Then use: 'q' to quit | 'c' to clear sentence | 'u' to undo last word")

    sentence = []
    last_label = None
    cooldown_frames = 20
    frame_counter = cooldown_frames
    max_sentence_length = 15
    forming_sentence = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error capturing frame.")
            break

        # --- CORREÇÃO: extrair apenas o pictograma ---
        roi_square, frame_with_rect = extract_pictogram_roi(frame)
        found_this_frame = False

        if roi_square is not None:
            resized = cv2.resize(roi_square, image_size)
            normalized = resized.astype('float32') / 255.0
            input_image = normalized.reshape(1, 64, 64, 1)

            predictions = model.predict(input_image, verbose=0)
            class_index = np.argmax(predictions)
            confidence = np.max(predictions)

            if confidence > 0.70:
                label = class_names[class_index]
                text = f"{label} ({confidence * 100:.1f}%)"

                # CORREÇÃO: Obter coordenadas do contorno já calculado na função extract_pictogram_roi
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                thresh = cv2.adaptiveThreshold(
                    blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7
                )
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Verificar se existem contornos antes de tentar encontrar o máximo
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Mostra label no topo do retângulo
                    cv2.putText(frame_with_rect, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    # Se não há contornos, mostrar o texto no canto superior esquerdo
                    cv2.putText(frame_with_rect, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if forming_sentence:
                    if (label != last_label or frame_counter >= cooldown_frames) and len(
                            sentence) < max_sentence_length:
                        sentence.append(label)
                        last_label = label
                        frame_counter = 0
                        found_this_frame = True

                cv2.imshow("ROI", resized)

        if not found_this_frame:
            frame_counter += 1

        displayed_sentence = sentence[-max_sentence_length:]
        full_sentence = " ".join(displayed_sentence)
        if len(sentence) > max_sentence_length:
            full_sentence = "... " + full_sentence

        header_text = f"Sentence: {full_sentence}" if forming_sentence else "Press 's' to start the sentence"
        cv2.putText(frame_with_rect, header_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow('Pictogram Identification - Q: quit | C: clear | U: undo | S: start', frame_with_rect)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            sentence = []
            last_label = None
            frame_counter = cooldown_frames
            print("Clear")
        elif key == ord('u'):
            if sentence:
                removed = sentence.pop()
                print(f"Last pictogram removed: {removed}")
                last_label = sentence[-1] if sentence else None
                frame_counter = cooldown_frames
        elif key == ord('s'):
            forming_sentence = True
            print("Sentence started")

    cap.release()
    cv2.destroyAllWindows()


# === MAIN MENU ===

if __name__ == '__main__':
    while True:
        #augment_data()
        #train_model()
        start_camera_prediction()