import os
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight
import random

#  CONFIGURAÇÕES
DATA_PATH = "./dataset_OrganizedSismicos_final"
MODEL_SAVE_PATH = './model_checkpoint/resnet50_sismico_pro.pth'

# Parâmetros
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 30  
PATIENCE = 5    
NUM_WORKERS = 2
SEED = 42

# Hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

set_seed(SEED)
print(f"🔧 Dispositivo: {DEVICE}")

# CARREGAMENTO DE DADOS
stats = ((0.5,), (0.5,))

# Augmentation "Leve" (Online) - Já fizemos o pesado no disco
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5), # Rotação leve extra
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

# Validação/Teste (Puro)
transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

print("\n📂 Carregando Datasets...")

# Verifica pastas
for split in ['train', 'validation', 'test']:
    path = os.path.join(DATA_PATH, split)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pasta '{path}' não encontrada! Rode o script de geração de dados primeiro.")

train_dataset = datasets.ImageFolder(os.path.join(DATA_PATH, 'train'), transform=transform_train)
val_dataset = datasets.ImageFolder(os.path.join(DATA_PATH, 'validation'), transform=transform_val)
test_dataset = datasets.ImageFolder(os.path.join(DATA_PATH, 'test'), transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                        num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                         num_workers=NUM_WORKERS, pin_memory=True)

class_names = train_dataset.classes
print(f"Classes: {class_names}")
print(f"Imagens: Treino={len(train_dataset)} | Val={len(val_dataset)} | Teste={len(test_dataset)}")

# 3. MODELO
class SeismicResNet(nn.Module):
    def __init__(self, num_classes):
        super(SeismicResNet, self).__init__()
        # Pesos V2 são mais modernos e precisos que os V1
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Congela camadas iniciais
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Descongela Layer 3 e 4 para refinar features de médio/alto nível
        for param in self.resnet.layer3.parameters():
            param.requires_grad = True
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

# 4. TREINAMENTO
def treinar():
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    model = SeismicResNet(len(class_names)).to(DEVICE)
    
    # Como o dataset agora é balanceado (5000 cada), não precisamos de pesos agressivos.
    # Usamos LabelSmoothing para evitar que o modelo fique "confiante demais" e erre feio.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-3)
    
    # CORREÇÃO: verbose=True removido
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print("\n🚀 INICIANDO TREINAMENTO...")

    for epoch in range(NUM_EPOCHS):
        # --- TREINO ---
        model.train()
        running_loss = 0.0
        running_correct = 0
        
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{NUM_EPOCHS}")
        
        for inputs, labels in loop:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=DEVICE.type):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_correct += torch.sum(preds == labels.data).item()
            
            loop.set_postfix(loss=loss.item())
            
        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_acc = 100.0 * running_correct / len(train_dataset)
        
        # --- VALIDAÇÃO ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data).item()
                
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = 100.0 * val_correct / len(val_dataset)
        
        print(f"   Done: Train Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.2f}%")
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        scheduler.step(epoch_val_loss)

        # Print Manual do LR (substituindo verbose)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"   📉 LR Atual: {current_lr:.2e}")
        
        # Early Stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("   💾 Melhor modelo salvo!")
        else:
            patience_counter += 1
            print(f"   ⚠️ Sem melhoria ({patience_counter}/{PATIENCE})")
            
        if patience_counter >= PATIENCE:
            print("\n🛑 Early Stopping ativado!")
            break
            
    return model, history

# 5. AVALIAÇÃO
def avaliar_teste(model):
    print("\n🔍 Avaliando no conjunto de TESTE (Prova Real)...")
    
    # Carrega melhor modelo
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print("   ✅ Modelo carregado.")
    except:
        print("   ❌ Erro ao carregar modelo. Usando o atual.")

    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testando"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    print("\n--- Relatório Final ---")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusão Final')
    plt.savefig('matriz_final.png')
    plt.show()

def plotar_historico(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Treino')
    plt.plot(history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Treino')
    plt.plot(history['val_acc'], label='Val')
    plt.title('Acurácia')
    plt.legend()
    plt.savefig('graficos.png')
    plt.show()

if __name__ == '__main__':
    modelo, hist = treinar()
    plotar_historico(hist)
    avaliar_teste(modelo)
