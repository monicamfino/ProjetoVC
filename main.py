import cv2
import os

# Caminho para a pasta de imagens
PASTA_IMAGENS = 'imagens'

# Tamanho padrão das imagens (para normalização)
TAMANHO_IMAGEM = (64, 64)

def carregar_imagens_referencia():
    """Carrega todas as imagens da pasta e associa ao nome da classe"""
    referencias = {}
    for nome_arquivo in os.listdir(PASTA_IMAGENS):
        caminho = os.path.join(PASTA_IMAGENS, nome_arquivo)
        img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, TAMANHO_IMAGEM)
            nome_classe = nome_arquivo.replace('.png', '').lower()
            referencias[nome_classe] = img
    return referencias

def classificar_por_template(img_capturada, referencias):
    """Compara a imagem capturada com todas as imagens de referência"""
    melhores_resultados = {}
    for nome, ref in referencias.items():
        resultado = cv2.matchTemplate(img_capturada, ref, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(resultado)
        melhores_resultados[nome] = max_val

    melhor_classe = max(melhores_resultados, key=melhores_resultados.get)
    return melhor_classe, melhores_resultados[melhor_classe]

def capturar_com_camera_template():
    referencias = carregar_imagens_referencia()
    cap = cv2.VideoCapture(0)

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
            classe, score = classificar_por_template(imagem_redimensionada, referencias)
            print(f"🧠 Classificação: {classe} (confiança: {score:.2f})")

        elif tecla == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    capturar_com_camera_template()
