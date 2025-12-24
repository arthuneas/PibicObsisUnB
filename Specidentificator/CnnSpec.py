'''
-----------------------------
Redes Neurais Convolucionais para Classificação de Espectrogramas
-----------------------------
Este código foi adaptado para treinar um modelo que diferencia
imagens de espectrogramas e não-espectrogramas.
'''

import os 
import torch
import torchvision
import torchaudio
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# 1. PRÉ-PROCESSAMENTO DOS DADOS

# Vamos redimensionar para 64x64 e normalizar para 1 canal (escala de cinza).
IMG_SIZE = 64
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), # Garante que todas as imagens tenham 64x64 pixels
    transforms.Grayscale(num_output_channels=1), # Converte imagens para escala de cinza
    transforms.ToTensor(), # Converte a imagem para um Tensor PyTorch
    transforms.Normalize((0.5,), (0.5,)) # Normaliza para 1 canal (em vez de 3)
])


# 2. ARQUITETURA DA CNN 

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        #  O primeiro parâmetro (in_channels) agora é 1 para escala de cinza.
        self.conv1 = nn.Conv2d(1, 6, 5) # 1 canal de entrada, 6 de saída, filtro 5x5
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # O tamanho da camada linear precisa ser recalculado.
        # Para uma imagem de entrada 64x64:
        # 1. Conv1 (64 - 5 + 1) = 60 -> Pool (60 / 2) = 30
        # 2. Conv2 (30 - 5 + 1) = 26 -> Pool (26 / 2) = 13
        # O tamanho final é 16 (canais) * 13 * 13 (dimensão da imagem)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        
        # A camada de saída agora tem 2 neurônios, um para cada classe.
        self.fc3 = nn.Linear(84, 11)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Função para visualização
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 3. BLOCO DE EXECUÇÃO PRINCIPAL

if __name__ == '__main__':

    batch_size = 4
    num_epochs = 20 
    
    # Apontando para a sua pasta de dataset final.
    data_path = "./dataset_final"
    
    # Usando ImageFolder para carregar seus dados de treino.
    train_path = os.path.join(data_path, 'train')
    trainset = datasets.ImageFolder(root=train_path, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    # MUDANÇA: Usando ImageFolder para carregar seus dados de teste.
    test_path = os.path.join(data_path, 'test')
    testset = datasets.ImageFolder(root=test_path, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # MUDANÇA: Definindo suas classes.
    # O ImageFolder ordena as classes em ordem alfabética.
    classes = ('espectrograma', 'nao_espectrograma')

    # Visualização de algumas imagens de treino
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    fig = plt.figure(figsize=(20, 4))
    for idx in np.arange(batch_size):
        ax = fig.add_subplot(1, batch_size, idx + 1, xticks=[], yticks=[])
        img_to_show = images[idx] / 2 + 0.5
        npimg = img_to_show.numpy()
        # Adicionado cmap='gray' para exibir imagens em escala de cinza corretamente.
        plt.imshow(np.squeeze(npimg), cmap='gray')
        ax.set_title(classes[labels[idx]])
    plt.show()


    # Treinamento da Rede Neural 
    cnn = CNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

    print("Iniciando o treinamento...")

    # Calcula o número total de lotes (batches) em todo o treinamento
    total_lotes = len(trainloader) * num_epochs

    # Cria a barra de progresso UMA VEZ, antes de tudo
    progress_bar = tqdm(total=total_lotes, desc="Progresso Total do Treinamento", unit="lote")

    for epoch in range(num_epochs):
        # O loop interno sobre os lotes continua o mesmo
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = cnn(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Atualiza a barra de progresso em 1 passo a cada lote
            progress_bar.update(1)
            
            # Atualiza as informações exibidas na barra
            progress_bar.set_postfix(epoca=f'{epoch + 1}/{num_epochs}', perda=loss.item())

    # Fecha a barra de progresso ao final de tudo
    progress_bar.close()
    print('\nTreinamento finalizado.')

        # Salvando o modelo treinado
    model_path = './model/'
    file_name = 'espectrograma_cnn_model.pth' # Novo nome para o modelo
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    full_path = os.path.join(model_path, file_name)
    torch.save(cnn.state_dict(), full_path)
    print(f"Modelo salvo em: {full_path}")
    
    # Teste do Modelo 
    cnn.eval()
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    
    cnn.load_state_dict(torch.load(full_path))
    outputs = cnn(images)
    
    _, predicted = torch.max(outputs, 1)

    print("\n--- Exemplo de Previsão ---")
    print('Labels Reais:    ', ' '.join(f'{classes[labels[j]]:18s}' for j in range(batch_size)))
    print('Previsão do Modelo:', ' '.join(f'{classes[predicted[j]]:18s}' for j in range(batch_size)))
    
    # Análise da performance
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = cnn(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'\nAcurácia da rede no conjunto de teste: {accuracy:.2f} %')