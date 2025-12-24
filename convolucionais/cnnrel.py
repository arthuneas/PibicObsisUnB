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

#taxa de amostrgem em hz
fs = 1000
#periodo de 0 a 2 segundos
t = np.linspace(0, 2, 2 * fs)
#x é a juncao das senoides (sinais), sendo a funcão np.sin(parametros da função)
x = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)

plt.figure(figsize=(8, 4))
           
#Nesse método seguinte, temos parametros fundamentais para a criaçâo do espectrograma           
#O primeiro parametro (x) define a funcao/sinal trabalhada

#O segundo, NFFT define a quantidade de pontos processados por vez. 
#Maior NFFT, garantimos uma melhor resolução em frequência, pior resolução em tempo.
#Menor NFFT, garantimos uma melhor resolução em tempo, pior em frequência.

#O fs é a taxa de amostragem define a quantidade de amotras por segundo o sinal possui

#cmap é a coloração do espectrograma
plt.specgram(x, NFFT = 256, Fs = fs, cmap = 'viridis')
plt.xlabel("tempo (s)")
plt.ylabel("Frequência (Hz)")
plt.colorbar(label = "intensidade (dB)")
#caso eu queira estabelecer o foco em uma faixa especifica:
#plt.ylim(0, 200), aqui limitamos até 200hz

plt.savefig("espectrograma.png", dpi=300, bbox_inches='tight')
plt.show()




transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    batch_size = 4
    num_epochs = 20
    data_path = "./data"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    trainset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('aviao', 'carro', 'passaro', 'gato', 'veado', 'cachorro', 'sapo', 'cavalo', 'navio', 'caminhao')

    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    fig = plt.figure(figsize=(20, 4))
    for idx in np.arange(batch_size):
        ax = fig.add_subplot(1, batch_size, idx + 1, xticks=[], yticks=[])
        img_to_show = images[idx] / 2 + 0.5
        npimg = img_to_show.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(classes[labels[idx]])
    plt.show()

    cnn = CNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

    print("Iniciando o treinamento...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            
            optimizer.zero_grad()
            outputs = cnn(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'Época [{epoch + 1}/{num_epochs}, Lote {i + 1:5d}] | Perda: {running_loss / 2000:.3f}')
                running_loss = 0.0
                
    print('Treinamento finalizado.')

    model_path = './model/'
    file_name = 'cnn_model.pth'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    full_path = os.path.join(model_path, file_name)
    torch.save(cnn.state_dict(), full_path)
    print(f"Modelo salvo em: {full_path}")

    cnn.eval()
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    
    cnn.load_state_dict(torch.load(full_path))
    outputs = cnn(images)
    
    _, predicted = torch.max(outputs, 1)

    print("\n--- Exemplo de Previsão ---")
    print('Labels Reais:      ', ' '.join(f'{classes[labels[j]]:10s}' for j in range(batch_size)))
    print('Previsão do Modelo:', ' '.join(f'{classes[predicted[j]]:10s}' for j in range(batch_size)))

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
    print(f'\nAcurácia da rede nas 10000 imagens de teste: {accuracy:.2f} %')

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = cnn(images)
            _, predictions = torch.max(outputs, 1)
            
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