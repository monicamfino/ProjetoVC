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


# === PREDICTION WITH CAMERA (MULTIPLE PICTOGRAMS) ===

def load_class_names():
    return sorted(entry.name for entry in os.scandir(augmented_folder) if entry.is_dir())


def extract_multiple_pictogram_rois(frame):
    """
    Extrai múltiplos ROIs de pictogramas na imagem.
    Retorna uma lista de (roi, bbox) onde bbox = (x, y, w, h)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold adaptativo para separar pictogramas do fundo
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7
    )

    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pictogram_rois = []

    if not contours:
        return pictogram_rois

    # Filtrar e processar contornos válidos
    valid_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        # Filtros para identificar pictogramas válidos
        if (w >= 30 and h >= 30 and area >= 1000 and
                0.3 <= w / h <= 3.0):  # Ratio de aspecto razoável
            valid_contours.append((cnt, area, x, y, w, h))

    # Ordenar por área (maiores primeiro)
    valid_contours.sort(key=lambda x: x[1], reverse=True)

    # Processar os contornos válidos (máximo 5 pictogramas)
    for i, (cnt, area, x, y, w, h) in enumerate(valid_contours[:5]):
        # Verificar se não há sobreposição significativa com contornos já processados
        overlap = False
        for existing_roi, existing_bbox in pictogram_rois:
            ex, ey, ew, eh = existing_bbox
            # Verificar sobreposição
            if (x < ex + ew and x + w > ex and y < ey + eh and y + h > ey):
                overlap_area = (min(x + w, ex + ew) - max(x, ex)) * (min(y + h, ey + eh) - max(y, ey))
                if overlap_area > 0.3 * min(w * h, ew * eh):  # 30% de sobreposição
                    overlap = True
                    break

        if overlap:
            continue

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

        pictogram_rois.append((roi_square, (x, y, w, h)))

    return pictogram_rois


def predict_multiple_pictograms(model, frame, class_names):
    """
    Prediz múltiplos pictogramas numa imagem.
    Retorna lista de (label, confidence, bbox)
    """
    pictogram_rois = extract_multiple_pictogram_rois(frame)
    predictions = []

    for roi_square, bbox in pictogram_rois:
        # Redimensionar e normalizar
        resized = cv2.resize(roi_square, image_size)
        normalized = resized.astype('float32') / 255.0
        input_image = normalized.reshape(1, 64, 64, 1)

        # Fazer predição
        prediction = model.predict(input_image, verbose=0)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        if confidence > 0.70:  # Threshold de confiança
            label = class_names[class_index]
            predictions.append((label, confidence, bbox))

    return predictions


def capture_current_pictograms(predictions):
    """
    Captura as strings dos pictogramas detectados no momento atual
    """
    if not predictions:
        print("❌ Nenhum pictograma detectado no momento!")
        return []

    captured_labels = []
    print("=" * 50)

    # Ordenar por posição (esquerda para direita)
    sorted_predictions = sorted(predictions, key=lambda x: x[2][0])

    for i, (label, confidence, bbox) in enumerate(sorted_predictions):
        captured_labels.append(label)

    print(f"FRASE FORMADA: {' '.join(captured_labels)}")
    print("=" * 50 + "\n")

    return captured_labels


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

    print("\n" + "=" * 50)
    print("DETECTOR DE PICTOGRAMAS")
    print("=" * 50)
    print("INSTRUÇÕES:")
    print(" - Aponta os pictogramas para a câmara")
    print(" - 'S' para detetar a frase")
    print(" - 'Q' para sair")
    print("=" * 50 + "\n")

    # Cores diferentes para cada pictograma detectado
    colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

    # Lista para armazenar todas as capturas realizadas
    all_captures = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error capturing frame.")
            break

        # Detectar múltiplos pictogramas
        predictions = predict_multiple_pictograms(model, frame, class_names)

        # Desenhar retângulos e labels para cada pictograma detectado
        for i, (label, confidence, bbox) in enumerate(predictions):
            x, y, w, h = bbox
            color = colors[i % len(colors)]

            # Desenhar retângulo
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

            # Texto com label e confiança
            text = f"{label} ({confidence * 100:.1f}%)"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Mostrar instruções na tela
        cv2.putText(frame, "Clica 'S' para capturar",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Mostrar total de capturas realizadas
        capture_info = f"Capturas realizadas: {len(all_captures)}"
        cv2.putText(frame, capture_info, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow('Detector de Pictogramas - S: Capturar | Q: Sair', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # CAPTURAR PICTOGRAMAS ATUAIS
            captured = capture_current_pictograms(predictions)
            if captured:
                all_captures.append(captured)

    cap.release()
    cv2.destroyAllWindows()

# === MAIN MENU ===

if __name__ == '__main__':
        # augment_data()
        # train_model()
        start_camera_prediction()