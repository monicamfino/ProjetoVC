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

def combinar_imagens(nome_img1, nome_img2, referencias, modo='horizontal'):
    """Combina duas imagens de referência em uma única imagem."""
    img1 = referencias.get(nome_img1)
    img2 = referencias.get(nome_img2)

    if img1 is None or img2 is None:
        print("❌ Uma ou ambas as imagens não foram encontradas nas referências.")
        return None

    # Certifique-se de que ambas as imagens têm o mesmo tamanho
    img1 = cv2.resize(img1, TAMANHO_IMAGEM)
    img2 = cv2.resize(img2, TAMANHO_IMAGEM)

    # Combine as imagens
    if modo == 'horizontal':
        imagem_combinada = cv2.hconcat([img1, img2])
    elif modo == 'vertical':
        imagem_combinada = cv2.vconcat([img1, img2])
    else:
        print("❌ Modo inválido. Use 'horizontal' ou 'vertical'.")
        return None

    return imagem_combinada

def capturar_com_camera_template():
    referencias = carregar_imagens_referencia()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Erro ao abrir a câmera.")
        return

    print("📷 Pressione 'c' para capturar e classificar | 'm' para combinar imagens | 'q' para sair")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Falha ao capturar imagem da câmera.")
            break

        # Espelhar o frame horizontalmente
        frame = cv2.flip(frame, 1)

        cv2.imshow('Câmera', frame)
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord('c'):
            imagem_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            imagem_redimensionada = cv2.resize(imagem_cinza, TAMANHO_IMAGEM)
            classe, score = classificar_por_template(imagem_redimensionada, referencias)
            print(f"🧠 Classificação: {classe} (confiança: {score:.2f})")

        elif tecla == ord('m'):
            # Exemplo de combinação de imagens
            nome_img1 = input("Digite o nome da primeira imagem: ").strip().lower()
            nome_img2 = input("Digite o nome da segunda imagem: ").strip().lower()
            imagem_combinada = combinar_imagens(nome_img1, nome_img2, referencias, modo='horizontal')

            if imagem_combinada is not None:
                cv2.imshow('Imagem Combinada', imagem_combinada)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        elif tecla == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    capturar_com_camera_template()