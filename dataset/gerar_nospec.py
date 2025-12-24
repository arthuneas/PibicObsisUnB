import tensorflow as tf
import os
import cv2 

# CONFIGURAÇÃO
OUTPUT_CIFAR10_FOLDER = './dataset/nao_espectrograma/cifar10'
# FIM DA CONFIGURAÇÃO 

# Cria a pasta de saída
if not os.path.exists(OUTPUT_CIFAR10_FOLDER):
    print(f"Criando a pasta de saída em: {OUTPUT_CIFAR10_FOLDER}")
    os.makedirs(OUTPUT_CIFAR10_FOLDER)
else:
    print(f"A pasta de saída já existe em: {OUTPUT_CIFAR10_FOLDER}")

try:
    print("--- Tentando baixar o dataset CIFAR-10... (Isso pode demorar na primeira vez) ---")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    print("--- Download/carregamento do CIFAR-10 concluído com sucesso! ---")

    all_images = tf.concat([x_train, x_test], axis=0)
    print(f"--- Total de imagens carregadas: {len(all_images)} ---")

    print(f"--- Começando a salvar as imagens em {OUTPUT_CIFAR10_FOLDER}... ---")

    # Itera e salva cada imagem
    for i, image_array in enumerate(all_images):
        if i % 5000 == 0:
            print(f"Salvando imagem {i} de {len(all_images)}...")
        
        # Converte a imagem de Tensor para NumPy array ANTES de passar para o OpenCy
        image_numpy = image_array.numpy()
        
        # Converte de RGB para BGR para salvar corretamente com OpenCV
        image_bgr = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
        
        # Define o nome do arquivo
        filename = os.path.join(OUTPUT_CIFAR10_FOLDER, f'cifar10_img_{i}.png')
        
        # Salva a imagem
        cv2.imwrite(filename, image_bgr)
        
    print(f"--- FIM: {len(all_images)} imagens do CIFAR-10 salvas com sucesso! ---")

except Exception as e:
    print(f"\n!!!!!! OCORREU UM ERRO !!!!!!!")
    print(f"Erro: {e}")