import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
from concurrent.futures import ProcessPoolExecutor
import warnings

# Ignora avisos do Librosa sobre arquivos mp3 (se houver)
warnings.filterwarnings("ignore", category=UserWarning)

# CONFIGURAÇÃO
PASTA_ORIGEM_AUDIOS = r'C:\Users\arthur.almeida\Downloads\IRMAS-TrainingData'
PASTA_SAIDA_IMAGENS = './dataset_instrumentos_otimizado' 
VERSOES_POR_AUDIO = 7
SR_PADRAO = 22050

FIG_SIZE_CONSISTENTE = (2.24, 2.24) # Proporção fixa para todas as imagens
CMAP_CONSISTENTE = 'magma'    # Um único mapa de cor de alto contraste

# === FUNÇÕES ===

def gerar_aumentos_audio(y, sr):
    """
    Aplica aumentos de dados de forma aleatória e controlada.
    (Esta é a sua função, está ótima!)
    """
    # Pitch shift leve (±2 semitons)
    if random.random() < 0.7:
        n_steps = random.uniform(-2.0, 2.0)
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

    # Time stretch leve (±15%)
    if random.random() < 0.7:
        rate = random.uniform(0.85, 1.15)
        y = librosa.effects.time_stretch(y, rate=rate)

    # Ruído branco proporcional
    if random.random() < 0.8:
        noise_amp = 0.005 * np.random.uniform() * np.amax(y)
        # Garante que y não seja somente zeros
        if np.amax(y) > 0:
            y = y + noise_amp * np.random.normal(size=y.shape)

    # Random crop (remove partes iniciais/finais)
    if len(y) > sr * 3 and random.random() < 0.6: # 3 segundos de áudio
        start = random.randint(0, int(len(y) * 0.1))
        end = random.randint(int(len(y) * 0.9), len(y))
        y = y[start:end]

    return y


def salvar_espectrograma_consistente(y, sr, caminho_saida):
    # Garante que o áudio não esteja silêncioso (evita erros)
    if np.abs(y).max() == 0:
        return

    S = librosa.feature.melspectrogram(y = y, sr=sr, n_mels=224)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Usa os parâmetros consistentes definidos no início
    plt.figure(figsize=FIG_SIZE_CONSISTENTE)
    librosa.display.specshow(S_db, sr=sr, cmap=CMAP_CONSISTENTE)

    # SALVA A IMAGEM "LIMPA" - SEMPRE sem eixos, bordas ou barras de cor
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(caminho_saida, dpi=100, bbox_inches='tight', pad_inches=0) # dpi=100 é suficiente
    
    plt.close()


def processar_audio(caminho_audio, pasta_destino):
    # Processa um único arquivo de áudio gerando múltiplos espectrogramas.
    try:
        y, sr = librosa.load(caminho_audio, sr=SR_PADRAO)
        nome_base = os.path.splitext(os.path.basename(caminho_audio))[0]

        for i in range(VERSOES_POR_AUDIO):
            # v1 (i=0) é o original, os outros são aumentados
            y_mod = gerar_aumentos_audio(y, sr) if i > 0 else y 
            caminho_saida = os.path.join(pasta_destino, f"{nome_base}_v{i+1}.png")
            
            salvar_espectrograma_consistente(y_mod, sr, caminho_saida) 

    except Exception as e:
        print(f"[ERRO] {caminho_audio}: {e}")


# === EXECUÇÃO PRINCIPAL ===
if __name__ == '__main__':
    if not os.path.exists(PASTA_ORIGEM_AUDIOS):
        print(f"ERRO: Pasta '{PASTA_ORIGEM_AUDIOS}' não encontrada!")
        exit()

    os.makedirs(PASTA_SAIDA_IMAGENS, exist_ok=True)

    # Coleta todos os caminhos de áudio e prepara destinos
    tarefas = []
    for nome_instrumento in os.listdir(PASTA_ORIGEM_AUDIOS):
        pasta_origem = os.path.join(PASTA_ORIGEM_AUDIOS, nome_instrumento)
        if not os.path.isdir(pasta_origem):
            continue

        pasta_destino = os.path.join(PASTA_SAIDA_IMAGENS, nome_instrumento)
        os.makedirs(pasta_destino, exist_ok=True)

        for f in os.listdir(pasta_origem):
            if f.lower().endswith(".wav"):
                tarefas.append((os.path.join(pasta_origem, f), pasta_destino))

    print(f"Total de áudios detectados: {len(tarefas)}")
    print(f"Total de imagens a serem geradas: {len(tarefas) * VERSOES_POR_AUDIO}")

    # Desempacota a lista de tarefas em listas de argumentos separadas
    # Isso evita o uso da 'lambda' que não pode ser "pickled"
    caminhos_audios = [t[0] for t in tarefas]
    pastas_destino = [t[1] for t in tarefas]
    
    num_workers = max(1, os.cpu_count() - 2)
    print(f"Usando {num_workers} núcleos para processamento...")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        
        # Agora passamos a função e as listas de argumentos separadamente
        # O map vai executar: processar_audio(caminhos_audios[0], pastas_destino[0])
        map_iterator = executor.map(processar_audio, caminhos_audios, pastas_destino)
        
        list(tqdm(map_iterator,
                  total=len(tarefas),
                  desc="Gerando espectrogramas"))

    print(f"\nPROCESSO COMPLETO! Dataset salvo em '{PASTA_SAIDA_IMAGENS}'")