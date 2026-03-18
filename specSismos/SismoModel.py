import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import random

#  CONFIGURAÇÕES
# Verifica se o dataset rápido existe
if os.path.exists('/content/dataset_sismico_temp'):
    DATA_PATH = '/content/dataset_sismico_temp'
    print("Dataset encontrado na memória do Colab!")
else:
    # Se você fez upload manual ou montou drive
    DATA_PATH = "./dataset_OrganizedSismicos_final"
    print("Dataset local ou não encontrado na memória temporária.")

# Salva no Drive
MODEL_SAVE_PATH = '/content/drive/MyDrive/PibicOasis/efficientnet_b0_focal.pth'
# Fallback local se o drive não estiver montado
if not os.path.exists(os.path.dirname(MODEL_SAVE_PATH)):
    MODEL_SAVE_PATH = './efficientnet_b0_focal.pth'

print(f"📂 Dados: {DATA_PATH}")
print(f"💾 Modelo será salvo em: {MODEL_SAVE_PATH}")

IMG_SIZE = 224 # Tamanho nativo da EfficientNet-B0
BATCH_SIZE = 32
NUM_EPOCHS = 30  
PATIENCE = 8     
NUM_WORKERS = 2
SEED = 42
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
if not os.path.exists(DATA_PATH) or not os.path.exists(os.path.join(DATA_PATH, 'train')):
    raise FileNotFoundError("Dataset não encontrado! Rode o script GERADOR primeiro.")

stats = ((0.5,), (0.5,))

transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15), 
    # Shear ajuda a simular distorções na onda
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10), 
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

print("\n Carregando Loaders")
train_dataset = datasets.ImageFolder(os.path.join(DATA_PATH, 'train'), transform=transform_train)
val_dataset = datasets.ImageFolder(os.path.join(DATA_PATH, 'validation'), transform=transform_val)
test_dataset = datasets.ImageFolder(os.path.join(DATA_PATH, 'test'), transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

class_names = train_dataset.classes
print(f"Classes: {class_names}")

# FOCAL LOSS
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=0.1)
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * CE_loss
        return torch.mean(F_loss) if self.reduction == 'mean' else F_loss

# MODELO: EFFICIENTNET-B0
class SeismicEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(SeismicEfficientNet, self).__init__()
        # Carrega a B0 pré-treinada
        self.effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Descongela tudo (Fine-Tuning total)
        for param in self.effnet.parameters():
            param.requires_grad = True 

        # A camada final da EfficientNet fica dentro de .classifier
        # O índice [1] geralmente é a Linear final original. Pegamos o in_features dela.
        num_ftrs = self.effnet.classifier[1].in_features
        
        # Nova cabeça personalizada
        self.effnet.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_ftrs, 512),
            nn.SiLU(), # EfficientNet usa SiLU (Swish)
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.effnet(x)


# TREINAMENTO
def treinar():
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    model = SeismicEfficientNet(len(class_names)).to(DEVICE)
    criterion = FocalLoss(gamma=2.5).to(DEVICE) 
    
    # EfficientNet gosta de Learning Rates um pouco maiores que ResNet no início
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

    best_val_acc = 0.0 
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print("\n INICIANDO TREINAMENTO (EFFICIENTNET)...")

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
            
            scheduler.step(epoch + loop.n / len(train_loader))
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
        
        print(f"   Done: Train Acc: {epoch_train_acc:.2f}% | Val Acc: {epoch_val_acc:.2f}% | Val Loss: {epoch_val_loss:.4f}")
        
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   Melhor modelo salvo! (Acc: {best_val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"   Sem melhora ({patience_counter}/{PATIENCE})")
            
        if patience_counter >= PATIENCE:
            print("Early Stopping!")
            break

    return model, history


def avaliar_teste(model):
    print("\nAvaliando Teste")
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print("   Checkpoint carregado.")
    
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    print("\n--- Relatório Final ---")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Matriz de Confusão (EfficientNet-B0)")
    plt.show()

if __name__ == '__main__':
    modelo, hist = treinar()
    avaliar_teste(modelo)
