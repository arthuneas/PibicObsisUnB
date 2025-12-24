import torch
import torch.nn as nn #camadas de redes
import torch.nn.functional as F #funcoes de ativação
import torch.optim as optim #otimizadores

import matplotlib.pyplot as plt #gera o grafico com o numero esperado

from PIL import Image #importa imagem 

from torchvision import datasets, transforms 
from torch.utils.data import DataLoader
#transforms transforma imagens em tensores 

input = str(input('DIGITE O CAMINHO DA IMAGEM: ')).strip()#link da imagem
imagem = Image.open(input) #abre a imagem de qualquer tamanho

######################################################################################################################
#transforma a imagem em um vetor multidimensional e normaliza cada dado em valores entre 0 e 1.
transform = transforms.Compose([
    transforms.Grayscale(), #garante a cor da imagem
    transforms.Resize((28,28)), #redimensiona a imagem de qualquer tamanho para 28x28 de dimensão
    transforms.ToTensor(), #trasnforma a imagem para tensor
    transforms.Normalize((0.5,),(0.5,)) #media e desvio = 0.5
])

######
#datasets de treino e teste
img_tensor = transform(imagem)
img_tensor = img_tensor.unsqueeze(0)  # adiciona dimensão de batch
train_dataset = datasets.MNIST(root = './data', train = True, download = True, transform = transform)
test_dataset = datasets.MNIST(root = "./data", train = False, download = True, transform = transform)
"""
metodo 1 - datasets.MNIST(1, 2, 3, 4)
-> o MNIST pega os arquivos sem precisar baixá-lo manualmente
1 - cria ou procurar os arquivos do dataset
2 - train = True, dados que a imagem usa para aprender.
3 - se os arquivos do MNIST nâo estiverem na pasta root, o torch baixa da internet automaticamente. Não baixa se já estiver no arquivo.
4 - toda imagem carregada 

metodo 2 - datasets.MNIST(1, 2, 3, 4)
-> diferença: train = Falsee
2 - pega os dados de teste e observa se a rede aprendeu de verdade
"""

######
train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False)
"""
DataLoader(1, 2, 3, 4)
nesse método, fazemos iterações sobre o datasett. percorremos os dados em lotes(batches) sem precisar escrever loops para acessar cada imagem
como parâmetros teremos :

1 - pega o dataset que você quer carregar, sendo o MNIST de treino
2 - treina em grupos de 64 imagens, fazemo
3 - numeros de exemplos que o DataLoader retorna em cada iteração, 
tendo o Dataset 60000, o tamanho do batch sendo 64, 60000/64 = 938 iterações por epoch.  
controla a quantidade de imagens que a rede processa antes de atualizar os pesos
4 - embaralha a ordem das imagens antes de cada época. evitamos uma rede fixa, logo, evitamos a generalização e o vicio da rede.
    -> shuffle = false, no test_loader nâo é necessário o embaralhamento.


"""
######################################################################################################################
#definir uma rede neural
class MLP(nn.Module):
    #toda rede usa nn.Module como parâmetro
    def __init__(self):
        super(MLP, self).__init__()
        # camadas lineares
        self.fc1 = nn.Linear(28*28, 128)  # primeira camada oculta, pega 784 pixels e reduz para 128 neuronios
        self.fc2 = nn.Linear(128, 64)     # segunda camada oculta, pega os 128 neuronios e reduz para 68 neuronios
        self.fc3 = nn.Linear(64, 10)      # saída: 10 classes (0-9), faz de 64 dimensoes para 10 saidas.

    def forward(self, x):
        x = x.view(-1, 28*28)  # achata a imagem em um vetor 
        x = F.relu(self.fc1(x)) #a função relu garante a complexiade da rede, com as formas e curvas sendo aprendidas
        x = F.relu(self.fc2(x)) #como self.fc1 e fc2 armazenam neurônios, estes aprendem as complexidades.
        x = self.fc3(x)         # logits, sem softmax, pois CrossEntropyLoss já aplica
        return x #retorno: retorna um número para cada classe de 0 a 9, ou seja, a classe com o maior valor é a saida correta
    
    
    
######################################################################################################################
model = MLP() #cria uma instancia da rede MLP
criterion = nn.CrossEntropyLoss() # essa é a função de perda, indica para a rede o seu erro em cada previsão
optimizer = optim.Adam(model.parameters(), lr=0.001) 
#adam é um bom otimizador, visto que não necessário o ajuste de muitos parâmetros. ajusta pesos da rede com base no gradient
"""
optim.Adam(1, 2)
1 -> model.parameters(): aqui selecionamos automaticamente todos os pesos da rede como parametros
2 -> lr: learning rate, controla a velocidade de aprendizado da rede, controla quanto do peso deve modificar cada iteraçâo.
"""



######################################################################################################################
#loop de treinamento da rede
num_epochs = 5 #aqui definimos o número de épocas

for epoch in range(num_epochs): #cada iteração é uma epoca
    for images, labels in train_loader: #aqui percorremos os dados nos batches até o final, ou seja, 64 imagens e 64 labels de uma vez
        optimizer.zero_grad()  #zera os gradientes acumulados na iteraçao anterior, para sabermos o valor da interação atual do gradiente, precisa-se zerá-lo a cada interação, já que ele, por padrão, será somado.
        outputs = model(images) #aqui a rede analisa cada imagem e ranqueia qual número parece ser mais próximo de ser o esperado.
        loss = criterion(outputs, labels)  # aqui temos a comparaçao do output da rede com o resultado esperado, quando maior o erro, maior o valor do criterion, aqui calculamos a perda.
        loss.backward()   #aqui o pytorch após a analie do erro, faz o caminho inverso na rede e averigua qual ponto contribuiu para o erro, sendo guardados como gradientes. Ou seja, a rede aprende com o erro.
        optimizer.step()  # aqui o omimizador ADAM, usado previamente, faz os ajustes da rede de acordo com os erros
    
    print(f'Época {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}') #aqui mostra a epoca e quanto a rede errou na epoca
    
    
    
######################################################################################################################
correct = 0 #conta a quantidade de previsões acertadas
total = 0 #conta a quantidade de imagens testadas
model.eval() #coloca a rede em modo
with torch.no_grad(): #o pytorch desliga o gradiente, pois estamos fazendo teste e exibição, nâo treino. Economiza memoria e tempo
    for images, labels in test_loader: #pega os grupo de imagens, o batches, e as saidas corretas, as labels od conjunto de teste
        outputs = model(images) #analise da rede para cada digito e garante oss scores para cada digito
        _, predicted = torch.max(outputs.data, 1) #pega o maior valor por imagem e o indice desse valor é o digito previsto
        total += labels.size(0) #soma a quantidade de imagens do teste até o total do MNIST de teste.
        correct += (predicted == labels).sum().item() #compara a previsâo com o correto, acerto = True(1) e erro = False(0), e tudo soma ao correct
  
acc = 100 * correct / total
print(f'TOTAL: {total:.0f}')
print(f'CORRECT: {correct:.0f}')
print(f'Taxa de acerto: {acc:.2f}%')

model.train()


#############################################################
#exibição
img_tensor, label = test_dataset[0] #pega a primera do conjunto de teste e da label
image_for_model = img_tensor.unsqueeze(0) #adiciona uma dimensao extra para a criação do batch de tamanho 1, pois a rede espera um batch, mesmo que seja uma unica imagem
 
output = model(image_for_model) #a imagem passa pela rede e gera logits(pontuação para cada digito)
_,predicted_digit = torch.max(output.data, 1) #pega o digito com maior pontuaçâo, ou seja, o digito que a rede percebeu

plt.imshow(img_tensor.squeeze(), cmap = 'gray')
plt.title(f"Previsto: {predicted_digit.item()}, Real: {label}") #exibe no titulo o numero esperado pela rede e o numero 
plt.axis('off') #retira eixos(x, y, z)
plt.show() #mostra a imagem no gráfico