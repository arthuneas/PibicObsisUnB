import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os
import librosa
import numpy as np
import io 
import matplotlib.pyplot as plt
import librosa.display

# ARQUITETURA DO MODELO
class FineTunedResNet(nn.Module):
    def __init__(self, num_classes):
        super(FineTunedResNet, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True 

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),  
            nn.Linear(num_ftrs, num_classes)
        )
    def forward(self, x):
        return self.resnet(x)


# FUNÇÃO DE PREVISÃO 
def classificar_audio(caminho_do_audio, modelo, classes, device):
    
    SR_PADRAO = 22050
    FIG_SIZE_CONSISTENTE = (2.24, 2.24) 
    CMAP_CONSISTENTE = 'magma'
    
    try:
        y, sr = librosa.load(caminho_do_audio, sr=SR_PADRAO)
        
        if np.abs(y).max() == 0:
            return "Áudio silencioso", 0

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=224)
        S_db = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=FIG_SIZE_CONSISTENTE)
        librosa.display.specshow(S_db, sr=sr, cmap=CMAP_CONSISTENTE)
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close() 
        buf.seek(0) 
        
        imagem_pil = Image.open(buf).convert('RGB')
        buf.close() 

        # Pré-processamento 
        IMG_SIZE = 224 
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),           
            
            transforms.Grayscale(num_output_channels=3), 
            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        imagem_tensor = transform(imagem_pil).unsqueeze(0).to(device) 

        # Inferência
        modelo.eval() 
        with torch.no_grad():
            output = modelo(imagem_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        previsao = classes[predicted_idx.item()]
        confianca = confidence.item() * 100

        return previsao, confianca

    except FileNotFoundError:
        return f"Erro: O arquivo de áudio não foi encontrado em '{caminho_do_audio}'", 0
    except Exception as e:
        return f"Ocorreu um erro ao processar o áudio: {e}", 0


# EXECUÇÃO PRINCIPAL 
if __name__ == '__main__':
    
    MODELO_PATH = r'C:\Users\arthur.almeida\Downloads\resnet50_instrumentos.pth' 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    if not os.path.exists(MODELO_PATH):
        print(f"ERRO: Checkpoint '{MODELO_PATH}' não encontrado.")
        exit()
    
    try:
        checkpoint = torch.load(MODELO_PATH, map_location=device)
        num_classes = checkpoint['num_classes']
        idx_to_class = {v: k for k, v in checkpoint['class_to_idx'].items()}
        classes_do_modelo = [idx_to_class[i] for i in range(num_classes)] 

        resnet_carregada = FineTunedResNet(num_classes=num_classes).to(device)
        resnet_carregada.load_state_dict(checkpoint['model_state_dict'])
        resnet_carregada.eval() 

        print(f"Modelo ResNet50 carregado! (Treinado por {checkpoint['epoch']} épocas, Acurácia: {checkpoint['best_val_acc']:.2f}%)")
        print(f"Classes mapeadas ({num_classes}): {classes_do_modelo}")

    except Exception as e:
        print(f"ERRO ao carregar o checkpoint: {e}")
        print("Verifique se a classe 'FineTunedResNet' está definida corretamente e se o checkpoint não está corrompido.")
        exit()

    while True:
        caminho_input = input("\nDIGITE O CAMINHO DO ARQUIVO DE ÁUDIO (ou 'sair'): ").strip('"')
        if caminho_input.lower() == 'sair':
            break

        previsao_instrumento, nivel_confianca = classificar_audio(caminho_input, resnet_carregada, classes_do_modelo, device)
        
        print("-" * 50)
        print(f"Analisando o áudio: {os.path.basename(caminho_input)}")
        if nivel_confianca > 0:
            print(f"--> Instrumento previsto: **{previsao_instrumento.upper()}** ({nivel_confianca:.2f}% de confiança)")
        else:
            print(f"--> {previsao_instrumento}")
        print("-" * 50)
