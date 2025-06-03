import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configurações do modelo e treino
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 40
IMAGES_FOLDER = 'images'
AUGMENTED_FOLDER = 'augmentation_images'
AUGMENTATION_PER_IMAGE = 30
MODEL_FILENAME = 'pictogram_cnn_model.keras'
CONFIDENCE_THRESHOLD = 0.70


def augment_data():
    #Aplica data augmentation às imagens originais
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

    os.makedirs(AUGMENTED_FOLDER, exist_ok=True)

    for file_name in os.listdir(IMAGES_FOLDER):
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(IMAGES_FOLDER, file_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Erro ao ler {img_path}. A ignorar...")
            continue

        img = cv2.resize(img, IMAGE_SIZE)
        img = img.reshape((IMAGE_SIZE[0], IMAGE_SIZE[1], 1))

        # Extrair label do nome do ficheiro
        label = os.path.splitext(file_name)[0].split('_')[0]
        label_dir = os.path.join(AUGMENTED_FOLDER, label)
        os.makedirs(label_dir, exist_ok=True)

        # Guardar imagem original
        original_save_path = os.path.join(label_dir, file_name)
        cv2.imwrite(original_save_path, img)

        # Criar imagens aumentadas
        for i, batch in enumerate(datagen.flow(np.array([img]), batch_size=1)):
            if i >= AUGMENTATION_PER_IMAGE:
                break

            augmented_img = batch[0].reshape(IMAGE_SIZE)
            new_name = f"{label}_{i:03d}.png"
            new_path = os.path.join(label_dir, new_name)
            cv2.imwrite(new_path, augmented_img)

    print("Data augmentation concluída!")


def create_model(num_classes):
    #Cria o modelo CNN
    return models.Sequential([
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
        layers.Dense(num_classes, activation='softmax')
    ])


def train_model():
    #Treina o modelo CNN
    datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        AUGMENTED_FOLDER,
        target_size=IMAGE_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        AUGMENTED_FOLDER,
        target_size=IMAGE_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # Carregar o modelo existente ou criar novo
    if os.path.exists(MODEL_FILENAME):
        print("A carregar o modelo existente...")
        model = tf.keras.models.load_model(MODEL_FILENAME)
    else:
        print("A criar novo modelo...")
        model = create_model(train_generator.num_classes)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("A treinar o modelo CNN...")
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS
    )

    model.save(MODEL_FILENAME)
    print(f"Modelo guardado como '{MODEL_FILENAME}'")


def load_class_names():
    #Carrega os nomes das classes.
    return sorted(entry.name for entry in os.scandir(AUGMENTED_FOLDER) if entry.is_dir())


def extract_pictogram_rois(frame):
    #Extrai ROIs de pictogramas da imagem
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    # Filtrar contornos válidos
    valid_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        if (w >= 30 and h >= 30 and area >= 1000 and 0.3 <= w / h <= 3.0):
            valid_contours.append((cnt, area, x, y, w, h))

    # Ordenar por área (maiores primeiro) e limitar a 5
    valid_contours.sort(key=lambda x: x[1], reverse=True)

    pictogram_rois = []
    for cnt, area, x, y, w, h in valid_contours[:5]:
        # Verificar sobreposição
        overlap = any(
            x < ex + ew and x + w > ex and y < ey + eh and y + h > ey and
            (min(x + w, ex + ew) - max(x, ex)) * (min(y + h, ey + eh) - max(y, ey)) > 0.3 * min(w * h, ew * eh)
            for _, (ex, ey, ew, eh) in pictogram_rois
        )

        if overlap:
            continue

        # Processar ROI
        roi = gray[y:y + h, x:x + w]

        # Tornar quadrado
        delta = abs(h - w)
        if h > w:
            left, right = delta // 2, delta - delta // 2
            top = bottom = 0
        else:
            top, bottom = delta // 2, delta - delta // 2
            left = right = 0

        roi_square = cv2.copyMakeBorder(roi, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)
        pictogram_rois.append((roi_square, (x, y, w, h)))

    return pictogram_rois


def predict_pictograms(model, frame, class_names):
    #Faz predict aos pictogramas na imagem
    pictogram_rois = extract_pictogram_rois(frame)
    predictions = []

    for roi_square, bbox in pictogram_rois:
        # Preprocessar
        resized = cv2.resize(roi_square, IMAGE_SIZE)
        normalized = resized.astype('float32') / 255.0
        input_image = normalized.reshape(1, 64, 64, 1)

        # Predict
        prediction = model.predict(input_image, verbose=0)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        if confidence > CONFIDENCE_THRESHOLD:
            label = class_names[class_index]
            predictions.append((label, confidence, bbox))

    return predictions


def capture_pictograms(predictions):
    #Captura e exibição os pictogramas detetados
    if not predictions:
        print("!!!NENHUM PICTOGRAMA DETETADO!!!")
        return []

    # Ordenar por posição horizontal
    sorted_predictions = sorted(predictions, key=lambda x: x[2][0])
    captured_labels = [label for label, _, _ in sorted_predictions]

    print("=" * 50)
    print(f"FRASE APRESENTADA: {' '.join(captured_labels)}")

    return captured_labels

def start_camera_prediction():
    #Inicia a predict pela câmera
    if not os.path.exists(MODEL_FILENAME):
        print("Modelo não encontrado. Treina o modelo primeiro.")
        return

    print("A carregar modelo...")
    model = tf.keras.models.load_model(MODEL_FILENAME)
    class_names = load_class_names()
    print(f"Classes carregadas: {class_names}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro ao abrir a câmara.")
        return

    print("\n" + "=" * 50)
    print("DETETOR DE PICTOGRAMAS")
    print("=" * 50)
    print("INSTRUÇÕES:")
    print(" - Aponta os pictogramas para a câmara")
    print(" - 'S' para capturar a frase")
    print(" - 'Q' para sair")
    print("=" * 50 + "\n")

    colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar frame.")
            break

        predictions = predict_pictograms(model, frame, class_names)

        # Desenhar deteções
        for i, (label, confidence, bbox) in enumerate(predictions):
            x, y, w, h = bbox
            color = colors[i % len(colors)]

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            text = f"{label} ({confidence * 100:.1f}%)"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(frame, "Clique 'S' para capturar", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow('Detetor de Pictogramas - S: Capturar | Q: Sair', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            capture_pictograms(predictions)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if not os.path.exists("pictogram_cnn_model.keras"):
        augment_data()
        train_model()
    start_camera_prediction()
    print("=" * 50 + "\n")