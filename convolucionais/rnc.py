'''
-----------------------------
Redes Neurais Convolucionais
-----------------------------
cnn - convolucional neural networks
trabalhada na classificação de imagens, majoritariamente.
através de um input, uma imagem, analisamos a imagem como uma matriz de imagens e passamos por uma funcao de ativação, com a relu e a convolução.
esse processo é realizdo até a consegurimos identificar os padrões da imagens, informaçoes que classficam a imagem.
após isso, passamos essa matriz convolucionada e reduzida à um vetor e passa pelas camadas da rede neural.
da rede passamos por uma funcao de softmax que calcula e fornece as probabilidades e classifica a imagem desejada.

- reconhecimento de padrao
exemplo: classificação da imagem de um cachorro.
através de um banco de dados com fotos de cachorros, um dataset, a rede identifica padrões entre as imagens, passa pelo clusters e as convoluçoes no processo.
uma imagem é representada como uma matriz de 3 canais (RGB), sendo assim, consegue-se formar diversas combinaçoes de cores a partir disso. porém, é utilizada uma escada grey.

a convolucao em si, é a analise da matriz fornecida, e uma matriz menor, faz essa validacao. Ele analisa o elemento similar da posição e multiplições e somas.
em resultancia, garantimos uma matriz convolucionada. ou seja, as convuluções são filtros que a analisam e comparam cada parte da imagem.

por exemplo, em analise de curvas, pegamos partes das imagens e passamos essa imagem por uma matriz com o filtro da curva, salientado que a imagem também é uma matriz com valores.
logo, a convolução garante a classificação e padronização da imagem. Inclusive, a convolução analisa com veemencia a intensidade de pixeis na imagem, o que amplifica

sobre os filtros:
podemos analisar a intensidade dos pixeis e com isso garantir uma padronização dos pixeis
pode-se analisar padroes horizontais e verticais da imagens, analisando curvas, linhas, etc.

nessa parte prática da criação da rede, foi usado um dataset Cifar10, o qual possui amplas imagens para classificação. O qual será utilizado para alimentar o modelo.
'''

import os 
import torch
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# pre processamento dos dados
# transformações da imagem para a rede 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Definição da arquitetura da rede neural
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # aqui temos a camada de convolução
        self.conv1 = nn.Conv2d(3, 6, 5) #nn.Conv2d(numero de dimensoes da imagens, numero de imagens de saida, tamanho da matriz do filtro 5x5)
        self.pool = nn.MaxPool2d(2, 2) # a camada de pool ajuda a apurar as caracteristicas mais importantes reduzindo a dimensao espacial pela metade, reduzindo a complexidade computacional
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # faremos que tenhamos 10 neuronios de saida, um para cada elemento da classe

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) #aqui garantimos que a rede passe pela camada convolucional fazendo o pooling, aplicando o relu
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # aqui achatamos o tensor 3d para um vetor, preservando a dimensao do batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) #nessa parte não ha necessidade de ativação, pois a lossfunction aplica a Softmax (probabilidades) internamente
        return x


# Função para visualização das imagens
# recebe oo tensor da imagem e exibe pelo matplotlib
def imshow(img):
    img = img / 2 + 0.5  # Desnormaliza a imagem
    npimg = img.numpy() #transforma a imagem para o formato esperado pelo Matplotlib, ou sejam, transforma o tensor para um array numpy
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# INÍCIO DO BLOCO DE EXECUÇÃO PRINCIPAL
# Todo o código que executa a lógica do programa vem aqui dentro
if __name__ == '__main__':

    batch_size = 4
    num_epochs = 20
    # garante a existencia e carregamento dos dados
    data_path = "./data"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    trainset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    # num_workers=2 é o que causa a necessidade do 'if __name__ == '__main__':'
    # os workers auxiliam no carregamento de dados para um processo otimizado
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('aviao', 'carro', 'passaro', 'gato', 'veado', 'cachorro', 'sapo', 'cavalo', 'navio', 'caminhao')

    # Visualização de algumas imagens de treino
    # pega um batch do dataloader de treino para visualização
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # cria uma figura do Matplotlib para exibir as imagens do lote
    fig = plt.figure(figsize=(20, 4))
    for idx in np.arange(batch_size):
        ax = fig.add_subplot(1, batch_size, idx + 1, xticks=[], yticks=[])
        img_to_show = images[idx] / 2 + 0.5
        npimg = img_to_show.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(classes[labels[idx]])
    plt.show()


    # Treinamento da Rede Neural
    cnn = CNN() #inicia 

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

    print("Iniciando o treinamento...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            
            optimizer.zero_grad() # zera o gradiente a cada periodo de treinamento para não ocorrer sobreposição da iteração anterior
            outputs = cnn(inputs) # passa os dados pela rede para obtenção de previsão
            loss = criterion(outputs, labels) # calcula o erro entre a previsão e o que é real
            loss.backward() # calcula os gradientes de perda
            optimizer.step() #direciona a rede para a direção em que minimiza a perda, ou seja, a derivada indo para zero na parábola
            
            # acumula e imprime a perda media a cada 2000 batches
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'Época [{epoch + 1}/{num_epochs}, Lote {i + 1:5d}] | Perda: {running_loss / 2000:.3f}')
                running_loss = 0.0
                
    print('Treinamento finalizado.')

    # Salvando o modelo treinado
    model_path = './model/'
    file_name = 'cnn_model.pth'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    full_path = os.path.join(model_path, file_name)
    torch.save(cnn.state_dict(), full_path)
    print(f"Modelo salvo em: {full_path}")

    # Testando o modelo com um lote de dados de teste
    cnn.eval()
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    
    # Carrega o modelo treinado
    cnn.load_state_dict(torch.load(full_path))
    outputs = cnn(images)
    
    # 'torch.max' retorna o valor máximo e o índice do máximo em cada linha
    # O índice corresponde à classe com a maior probabilidade
    _, predicted = torch.max(outputs, 1)

    print("\n--- Exemplo de Previsão ---")
    print('Labels Reais:    ', ' '.join(f'{classes[labels[j]]:10s}' for j in range(batch_size)))
    print('Previsão do Modelo:', ' '.join(f'{classes[predicted[j]]:10s}' for j in range(batch_size)))

    # Análise da performance geral do modelo
    correct = 0
    total = 0
    #desativa o gradiente posi o modelo está em teste, o que otimiza desempenho pela desocupaçãod e memoria
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = cnn(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'\nAcurácia da rede nas 10000 imagens de teste: {accuracy:.2f} %')

    # Análise da performance por classe
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = cnn(images)
            _, predictions = torch.max(outputs, 1)
            
            # compara a previsão com o modulo real para cada batch
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    print("\n--- Acurácia por Classe ---")
    for classname, correct_count in correct_pred.items():
        if total_pred[classname] > 0:
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Acurácia para a classe: {classname:10s} é {accuracy:.1f} %')
        else:
            print(f'Nenhuma imagem da classe {classname} foi encontrada no teste.')