import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Caminho para a pasta de imagens
PASTA_IMAGENS = 'imagens'

# Tamanho padrão das imagens (para normalizar)
TAMANHO_IMAGEM = (64, 64)

# Rótulos
rotulos = {'sim': 0, 'nao': 1, 'nao sei': 2}
rotulos_inverso = {v: k for k, v in rotulos.items()}


def carregar_imagem(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {path}")
    img = cv2.resize(img, TAMANHO_IMAGEM)
    return img.flatten()


def mostrar_imagem(path):
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()


def treinar_classificador():
    imagens_treino = [
        ('sim.png', 'sim'),
        ('nao.png', 'nao'),
        ('nao_sei.png', 'nao sei')
    ]

    X, y = [], []

    for nome_arquivo, rotulo in imagens_treino:
        caminho = os.path.join(PASTA_IMAGENS, nome_arquivo)
        vetor = carregar_imagem(caminho)
        X.append(vetor)
        y.append(rotulos[rotulo])

    X = np.array(X)
    y = np.array(y)

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X, y)
    return knn


def classificar_imagem(classificador, nome_imagem_teste):
    caminho = os.path.join(PASTA_IMAGENS, nome_imagem_teste)
    mostrar_imagem(caminho)
    vetor = carregar_imagem(caminho)
    pred = classificador.predict([vetor])[0]
    return rotulos_inverso[pred]


def capturar_com_camera(classificador):
    cap = cv2.VideoCapture(0)  # Usa a câmera padrão

    if not cap.isOpened():
        print("❌ Erro ao abrir a câmera.")
        return

    print("📷 Pressione 'c' para capturar e classificar | 'q' para sair")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Falha ao capturar imagem da câmera.")
            break

        cv2.imshow('Câmera', frame)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord('c'):
            imagem_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            imagem_redimensionada = cv2.resize(imagem_cinza, TAMANHO_IMAGEM)
            vetor = imagem_redimensionada.flatten()
            pred = classificador.predict([vetor])[0]
            resultado = rotulos_inverso[pred]
            print(f"🧠 Classificação: {resultado}")

        elif tecla == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    knn = treinar_classificador()

    modo = input("Digite 'c' para usar a câmera ou 'i' para classificar uma imagem: ").lower()

    if modo == 'c':
        capturar_com_camera(knn)
    elif modo == 'i':
        imagem_teste = input("Nome da imagem na pasta 'imagens/': ")
        resultado = classificar_imagem(knn, imagem_teste)
        print("📢 Resultado:")
        print(f"A imagem representa: {resultado}")
    else:
        print("❗ Opção inválida.")
