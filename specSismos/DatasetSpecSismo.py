import h5py
import os
import librosa
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import random
from PIL import Image
from scipy import signal 

ARQUIVO_H5 = r'C:\Users\arthur.almeida\Downloads\dados_sismicos.h5'
PASTA_RAIZ = './dataset_OrganizedSismicos_final' # Mudado para estrutura organizada
SAMPLE_RATE = 100 

# CONFIGURAÇÕES
META_TREINO_POR_CLASSE = 5000 # Meta apenas para o TREINO
SPLIT_VALIDACAO = 0.20        # 20% dos dados reais vão para validação (intocados)
IMG_SIZE = (224, 224)

# DATA AUGMENTATION (Igual ao anterior)
def gerar_variacao(sinal_in):
    sinal_mod = sinal_in.copy()
    if len(sinal_mod) < 200: return sinal_mod

    # Time Stretch
    if random.random() < 0.3:
        rate = np.random.uniform(0.9, 1.1)
        try:
            sinal_mod = librosa.resample(sinal_mod, orig_sr=SAMPLE_RATE, target_sr=int(SAMPLE_RATE*rate))
            if len(sinal_mod) > len(sinal_in):
                sinal_mod = sinal_mod[:len(sinal_in)]
            else:
                sinal_mod = np.pad(sinal_mod, (0, len(sinal_in) - len(sinal_mod)))
        except: pass

    # Shift
    if random.random() < 0.5:
        max_shift = int(len(sinal_in) * 0.15)
        shift_amt = np.random.randint(1, max_shift)
        if random.choice([True, False]):
            sinal_mod = np.pad(sinal_mod, (shift_amt, 0), mode='constant')[:-shift_amt]
        else:
            sinal_mod = np.pad(sinal_mod, (0, shift_amt), mode='constant')[shift_amt:]

    # Ruído
    if random.random() < 0.4:
        ruido = np.random.normal(0, 0.015 * np.max(np.abs(sinal_mod)), sinal_mod.shape)
        sinal_mod = sinal_mod + ruido

    # Amplitude
    fator = np.random.uniform(0.8, 1.2)
    sinal_mod = sinal_mod * fator

    return sinal_mod.astype(np.float32)

# WORKER COM DETREND + NORMALIZAÇÃO
def processar_tarefa(args):
    sinal_base, sr, caminho_final, aplicar_efeito = args
    
    # Remove viés DC ou tendências lineares do sensor antes de qualquer coisa
    sinal_base = signal.detrend(sinal_base)
    
    if aplicar_efeito:
        sinal_final = gerar_variacao(sinal_base)
    else:
        sinal_final = sinal_base
        
    sinal_final = sinal_final.astype(np.float32)

    try:
        hop_len = 32 
        
        S = librosa.feature.melspectrogram(
            y=sinal_final, sr=sr, n_fft=512, hop_length=hop_len, n_mels=128, fmax=sr/2
        )
        
        S_db = librosa.power_to_db(S, ref=np.max)
        S_db = np.maximum(S_db, -80) # Clamping no piso de ruído
        
        img_norm = (S_db + 80) / 80 * 255
        
        img_uint8 = img_norm.astype(np.uint8)
        img_uint8 = np.flipud(img_uint8)

        image = Image.fromarray(img_uint8)
        image = image.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        image = image.convert('L') 
        image.save(caminho_final)

    except Exception as e:
        print(f"Erro processando {caminho_final}: {e}")

# EXECUÇÃO COM SPLIT SEGURO
if __name__ == '__main__':
    print(f"--- INICIANDO ---")
    print(f"Meta Treino: {META_TREINO_POR_CLASSE} | Validação: {SPLIT_VALIDACAO*100}% dos reais")
    
    with h5py.File(ARQUIVO_H5, 'r') as f:
        X = f['X'][:]
        y = f['y'][:]
        classes_raw = f['classes'][:]
        nomes_classes = [c.decode('utf-8') if isinstance(c, bytes) else c for c in classes_raw]

    indices_por_classe = {}
    for i, nome in enumerate(nomes_classes):
        indices_por_classe[nome] = np.where(y == i)[0]

    tarefas = []
    
    # Cria pastas train e validation
    dir_train_root = os.path.join(PASTA_RAIZ, 'train')
    dir_val_root = os.path.join(PASTA_RAIZ, 'validation')
    os.makedirs(dir_train_root, exist_ok=True)
    os.makedirs(dir_val_root, exist_ok=True)

    for nome_classe, indices_originais in indices_por_classe.items():
        # Embaralha os índices originais para garantir aleatoriedade no split
        np.random.shuffle(indices_originais)
        
        total_originais = len(indices_originais)
        qtd_val = int(total_originais * SPLIT_VALIDACAO)
        
        # Índices reservados EXCLUSIVAMENTE para validação (nunca serão aumentados)
        idxs_val = indices_originais[:qtd_val]
        # Índices reservados para treino (serão aumentados para atingir a meta)
        idxs_train_seed = indices_originais[qtd_val:]
        
        # Preparar Validação (Apenas salva, sem augmentation)
        dir_classe_val = os.path.join(dir_val_root, nome_classe)
        os.makedirs(dir_classe_val, exist_ok=True)
        
        for i, idx in enumerate(idxs_val):
            caminho = os.path.join(dir_classe_val, f"{nome_classe}_val_{i}.png")
            tarefas.append((X[idx], SAMPLE_RATE, caminho, False)) # False = Sem efeito
            
        # Preparar Treino (Expande até a meta)
        dir_classe_train = os.path.join(dir_train_root, nome_classe)
        os.makedirs(dir_classe_train, exist_ok=True)
        
        qtd_seed = len(idxs_train_seed)
        if qtd_seed == 0: continue # Segurança
        
        # Lista de indices que vamos processar para treino
        # Se for NDA, cortamos se passar da meta. Se for outra classe, repetimos.
        indices_finais_treino = []
        
        if nome_classe == "NDA":
            # Para NDA, pegamos até a meta ou todos que sobraram
            limite = min(len(idxs_train_seed), META_TREINO_POR_CLASSE)
            indices_finais_treino = idxs_train_seed[:limite]
            usar_augment_neste_loop = False # NDA Treino também não recebe augment (geralmente)
        else:
            # Para classes raras, repetimos a lista seed até encher a meta
            for k in range(META_TREINO_POR_CLASSE):
                indices_finais_treino.append(idxs_train_seed[k % qtd_seed])
            usar_augment_neste_loop = True

        for i, idx in enumerate(indices_finais_treino):
            sinal = X[idx]
            
            # Lógica: Se é NDA, nunca aumenta.
            # Se é outra classe: As primeiras cópias (originais) não aumentam, as réplicas aumentam.
            if nome_classe == "NDA":
                aplicar = False
            else:
                # Se i < qtd_seed, estamos passando pelos originais pela primeira vez -> Manter puro
                # Se i >= qtd_seed, estamos nas cópias -> Aplicar augment
                aplicar = (i >= qtd_seed)
            
            caminho = os.path.join(dir_classe_train, f"{nome_classe}_train_{i}.png")
            tarefas.append((sinal, SAMPLE_RATE, caminho, aplicar))

    num_cores = max(1, os.cpu_count() - 2)
    print(f"\nProcessando {len(tarefas)} imagens em estrutura TRAIN/VAL...")
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        list(tqdm(executor.map(processar_tarefa, tarefas), total=len(tarefas)))

    print(f"\nCONCLUÍDO! Dataset pronto em: '{PASTA_RAIZ}'")
