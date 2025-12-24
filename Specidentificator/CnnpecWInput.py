import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt # Importar matplotlib para salvar a imagem processada
import numpy as np # Necessário para o matplotlib

# ARQUITETURA DA REDE (IDÊNTICA DO TREINO) 
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) 
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 13 * 13, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

# FUNÇÃO DE PREVISÃO 
def prever_imagem(caminho_da_imagem, modelo, classes, pasta_saida, salvar_processada=False):
    try:
        imagem_original = Image.open(caminho_da_imagem).convert('RGB')
    except FileNotFoundError:
        return f"Erro: O arquivo de imagem não foi encontrado em '{caminho_da_imagem}'"

    IMG_SIZE = 64
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    imagem_tensor = transform(imagem_original).unsqueeze(0)

    if salvar_processada:
        img_processada_np = imagem_tensor.squeeze(0).numpy() * 0.5 + 0.5
        
        nome_base = os.path.basename(caminho_da_imagem)
        nome_sem_ext = os.path.splitext(nome_base)[0]
        
        # <--- MUDANÇA: Usa os.path.join para criar o caminho de saída dentro da pasta especificada
        nome_arquivo_saida = f'{nome_sem_ext}_processada.png'
        caminho_saida = os.path.join(pasta_saida, nome_arquivo_saida)

        plt.imshow(np.transpose(img_processada_np, (1, 2, 0)), cmap='gray')
        plt.axis('off')
        plt.savefig(caminho_saida, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Imagem processada salva como: {caminho_saida}")

    modelo.eval()

    with torch.no_grad():
        output = modelo(imagem_tensor)
        _, predicted_idx = torch.max(output, 1)

    return classes[predicted_idx.item()]

# EXECUÇÃO PRINCIPAL
if __name__ == '__main__':
    MODELO_PATH = r'C:\Users\arthur.almeida\Downloads\Python\model\espectrograma_cnn_model.pth'
    classes_do_modelo = ('espectrograma', 'nao_espectrograma')
    
    # <--- MUDANÇA 1: Defina o nome da pasta onde as imagens serão salvas
    PASTA_SAIDA = 'imagens_processadas'
    
    # <--- MUDANÇA 2: Cria a pasta se ela não existir
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    
    cnn_carregada = CNN()

    if not os.path.exists(MODELO_PATH):
        print(f"ERRO: Modelo não encontrado em '{MODELO_PATH}'. Verifique se o caminho está correto.")
    else:
        cnn_carregada.load_state_dict(torch.load(MODELO_PATH))
        print(f"Modelo '{MODELO_PATH}' carregado com sucesso.")

        caminho_input = input("\nDIGITE O CAMINHO DA IMAGEM: ").strip('"')

        # <--- MUDANÇA 3: Passe o nome da pasta de saída para a função de previsão
        previsao = prever_imagem(caminho_input, cnn_carregada, classes_do_modelo, PASTA_SAIDA, salvar_processada=True)

        print("-" * 30)
        print(f"Analisando imagem: {os.path.basename(caminho_input)}")
        print(f"Resultado da Análise:")
        print(f"A imagem fornecida é: -> {previsao} <-")
        print("-" * 30)