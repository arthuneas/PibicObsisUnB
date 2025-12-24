import h5py
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import random

# 👇 CAMINHO CORRIGIDO
ARQUIVO_H5 = r'C:\Users\arthur.almeida\Downloads\dados_sismicos.h5'

PASTA_SAIDA = './dataset_sismos_final'
SAMPLE_RATE = 100 
META_POR_CLASSE = 7000  # Meta fixa de imagens por classe

FIG_SIZE = (2.24, 2.24)
CMAP = 'magma'
# ==========================================

def gerar_variacao(sinal):
    """Gera uma versão levemente diferente do sinal original."""
    tipo = random.choice(['ruido', 'shift', 'amplitude', 'inversao'])
    sinal_mod = sinal.copy()
    
    if tipo == 'ruido':
        ruido = np.random.normal(0, 0.035 * np.max(np.abs(sinal)), sinal.shape)
        sinal_mod = sinal + ruido
    elif tipo == 'shift':
        shift = np.random.randint(len(sinal) * 0.08)
        if random.random() > 0.5: shift = -shift
        sinal_mod = np.roll(sinal, shift)
    elif tipo == 'amplitude':
        fator = np.random.uniform(0.8, 1.2)
        sinal_mod = sinal * fator
    elif tipo == 'inversao':
        sinal_mod = -sinal

    return sinal_mod.astype(np.float32)

def processar_tarefa(args):
    sinal_base, sr, caminho_final, aplicar_efeito = args
    plt.ioff()
    
    # Aplica variação se for um clone de classe definida
    sinal_final = gerar_variacao(sinal_base) if aplicar_efeito else sinal_base
    sinal_final = sinal_final.astype(np.float32)

    try:
        # Gera Espectrograma
        S = librosa.feature.melspectrogram(
            y=sinal_final, sr=sr, n_fft=256, hop_length=64, n_mels=64
        )
        S_db = librosa.power_to_db(S, ref=np.max)

        # Salva Imagem Limpa
        fig = plt.figure(figsize=FIG_SIZE)
        librosa.display.specshow(S_db, sr=sr, cmap=CMAP)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(caminho_final, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    except Exception as e:
        print(f"Erro: {e}")

if __name__ == '__main__':
    # Verificação de arquivo
    if not os.path.exists(ARQUIVO_H5):
        print(f"ERRO CRÍTICO: Arquivo não encontrado em:\n{ARQUIVO_H5}")
        exit()

    print(f"--- INICIANDO COM META: {META_POR_CLASSE} IMAGENS POR CLASSE ---")
    
    # A. Leitura
    with h5py.File(ARQUIVO_H5, 'r') as f:
        X = f['X'][:]
        y = f['y'][:]
        classes_raw = f['classes'][:]
        nomes_classes = [c.decode('utf-8') if isinstance(c, bytes) else c for c in classes_raw]

    # B. Organização
    indices_por_classe = {}
    for i, nome in enumerate(nomes_classes):
        indices_por_classe[nome] = np.where(y == i)[0]

    tarefas = []
    os.makedirs(PASTA_SAIDA, exist_ok=True)

    # C. Planejamento das Tarefas
    for nome_classe, indices_originais in indices_por_classe.items():
        dir_classe = os.path.join(PASTA_SAIDA, nome_classe)
        os.makedirs(dir_classe, exist_ok=True)
        
        qtd_disponivel = len(indices_originais)
        
        # --- LÓGICA INTELIGENTE ---
        if nome_classe == "NDA":
            # NDA: Se tem mais que a meta, corta aleatoriamente. Não aplica Augmentation.
            if qtd_disponivel >= META_POR_CLASSE:
                indices_selecionados = np.random.choice(indices_originais, META_POR_CLASSE, replace=False)
                print(f"[{nome_classe}] Reduzindo de {qtd_disponivel} para {META_POR_CLASSE} (Apenas originais)")
            else:
                # Caso raro (se NDA fosse menor que 7000), completaria com originais repetidos
                indices_selecionados = [indices_originais[i % qtd_disponivel] for i in range(META_POR_CLASSE)]
                
        else:
            # Classes Definidas: Multiplica (clona) até atingir a meta
            indices_selecionados = [indices_originais[i % qtd_disponivel] for i in range(META_POR_CLASSE)]
            fator = META_POR_CLASSE / qtd_disponivel
            print(f"[{nome_classe}] Expandindo de {qtd_disponivel} para {META_POR_CLASSE} (Variação média: {fator:.1f}x)")

        # D. Criar lista de execução
        for i, idx_real in enumerate(indices_selecionados):
            sinal = X[idx_real]
            
            # Regra de Ouro para Efeitos:
            # 1. NDA -> Nunca aplica efeito (queremos o ruído original).
            # 2. Outros -> Aplica efeito apenas se for um clone (i >= qtd_original)
            if nome_classe == "NDA":
                aplicar_efeito = False
            else:
                aplicar_efeito = (i >= qtd_disponivel)
            
            nome_arquivo = f"{nome_classe}_{i}.png"
            caminho = os.path.join(dir_classe, nome_arquivo)
            tarefas.append((sinal, SAMPLE_RATE, caminho, aplicar_efeito))

    # E. Processamento
    num_cores = max(1, os.cpu_count() - 2)
    print(f"\nGerando {len(tarefas)} arquivos com {num_cores} núcleos...")
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        list(tqdm(executor.map(processar_tarefa, tarefas), total=len(tarefas)))

    print(f"\nSUCESSO! Confira a pasta: '{PASTA_SAIDA}'")