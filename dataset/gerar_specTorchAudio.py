import torch
import torchaudio
import matplotlib.pyplot as plt
import os
from gtts import gTTS
from tqdm import tqdm
import numpy as np
import random
import requests
import re
from PIL import Image 


print("--- Fábrica de Espectrogramas ---")
# 1. CONFIGURAÇÃO GERAL 

# Configuração da Fonte de Áudio
QUANTIDADE_DE_FRASES = 1000 # Quantas frases do livro queremos usar para gerar áudio
CAMINHO_LIVRO_TXT = './os_sertoes.txt' # Onde o livro será salvo
URL_LIVRO = 'https://www.gutenberg.org/cache/epub/5358/pg5358.txt' # Os Sertões, de Euclides da Cunha

# --- Configuração da Geração de Espectrogramas ---
VERSOES_POR_AUDIO = 5 # Quantas imagens diferentes gerar para cada arquivo de áudio
# Lista expandida de mapas de cores, incluindo mais opções vibrantes e inversas
LISTA_MAPAS_DE_CORES = [
    'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'gray_r', 'jet', 
    'rainbow', 'hsv', 'ocean', 'gist_earth', 'terrain', 'flag', 'prism',
    'cool', 'spring', 'summer', 'autumn', 'winter'
]

# Parâmetros para variar o MelSpectrogram (para diversificar a "textura")
N_FFT_OPTIONS = [1024, 2048, 4096]
HOP_LENGTH_OPTIONS = [256, 512, 1024]
N_MELS_OPTIONS = [64, 128, 256]

# --- Configuração das Pastas ---
PASTA_DE_AUDIOS = './audios_gerados_massivos'
PASTA_DE_SAIDA_ESPECTROGRAMAS = './espectrogramas_gerados_diversos'

# --- FIM DA CONFIGURAÇÃO ---


# 2. FUNÇÕES AUXILIARES 

def baixar_livro(url, caminho_salvar):
    """Baixa o arquivo de texto do livro se ele não existir."""
    if not os.path.exists(caminho_salvar):
        print(f"Baixando livro de {url}...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(caminho_salvar, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print("Download do livro concluído.")
        except requests.exceptions.RequestException as e:
            print(f"Erro ao baixar o livro: {e}")
            return False
    return True

def carregar_frases(caminho_livro, quantidade):
    """Carrega o livro, limpa e extrai um número de frases."""
    print("Carregando e processando frases do livro...")
    with open(caminho_livro, 'r', encoding='utf-8') as f:
        texto = f.read()
    texto = texto.replace('\n', ' ').replace('\r', '')
    frases = re.split(r'[.!?]+', texto)
    frases_limpas = [f.strip() for f in frases if len(f.strip()) > 15]
    random.shuffle(frases_limpas)
    return frases_limpas[:quantidade]

def gerar_arquivos_de_audio(lista_de_frases, pasta_saida):
    """Gera arquivos .mp3 a partir de uma lista de textos."""
    os.makedirs(pasta_saida, exist_ok=True)
    print(f"\n--- Etapa 1: Gerando {len(lista_de_frases)} arquivos de áudio ---")
    for i, texto in enumerate(tqdm(lista_de_frases, desc="Gerando Áudios")):
        try:
            tts = gTTS(text=texto, lang='pt-br')
            tts.save(os.path.join(pasta_saida, f"audio_{i+1}.mp3"))
        except Exception as e:
            print(f"Erro ao gerar áudio '{texto[:30]}...': {e}") # Mostra um pedaço do texto em caso de erro

# 3. FUNÇÃO PRINCIPAL DE GERAÇÃO DE ESPECTROGRAMAS

def gerar_espectrograma_de_audio(caminho_audio, caminho_saida, estilo_config):
    """
    Gera um espectrograma com opções de estilo (legendas, cor, duplicação, etc.).
    estilo_config: dict contendo 'com_legendas', 'mapa_de_cores', 'duplicar', 'n_fft', 'hop_length', 'n_mels'
    """
    try:
        waveform, sample_rate = torchaudio.load(caminho_audio)
        if waveform.ndim > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)

        transform_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, 
            n_fft=estilo_config['n_fft'], 
            hop_length=estilo_config['hop_length'], 
            n_mels=estilo_config['n_mels']
        )
        espectrograma_db = torchaudio.transforms.AmplitudeToDB()(transform_mel(waveform))

        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Converte para numpy para plotar
        spec_np = espectrograma_db.squeeze().numpy()

        if estilo_config['com_legendas']:
            # Versão COM legendas, eixos, título, etc.
            try:
                import librosa.display
                img = librosa.display.specshow(spec_np, sr=sample_rate, x_axis='time', y_axis='mel', 
                                              cmap=estilo_config['mapa_de_cores'], ax=ax)
                ax.set_title(f'Espectrograma de Mel', fontsize=10) # Título menor
                fig.colorbar(img, format='%+2.0f dB', ax=ax, shrink=0.75) # Barra de cores menor
                ax.tick_params(axis='x', labelsize=8) # Tamanho da fonte dos ticks
                ax.tick_params(axis='y', labelsize=8)
            except ImportError:
                ax.imshow(spec_np, cmap=estilo_config['mapa_de_cores'], origin='lower', aspect='auto')
                ax.set_xlabel('Tempo', fontsize=8)
                ax.set_ylabel('Frequência', fontsize=8)
                ax.set_title(f'Espectrograma', fontsize=10)
            plt.tight_layout(pad=0.5) # Ajusta o layout com um pouco de padding
            fig.savefig(caminho_saida, dpi=120) # Aumentei o DPI para melhor qualidade
        else:
            # Versão SEM legendas 
            ax.imshow(spec_np, cmap=estilo_config['mapa_de_cores'], origin='lower', aspect='auto')
            ax.axis('off')
            fig.savefig(caminho_saida, bbox_inches='tight', pad_inches=0, dpi=120)
        
        plt.close(fig)

        # Duplicar a imagem se configurado
        if estilo_config['duplicar']:
            img_original = Image.open(caminho_saida)
            width, height = img_original.size
            
            # Cria uma nova imagem com o dobro da altura para a duplicação
            img_duplicada = Image.new('RGB', (width, height * 2 + 5), color = 'black') # Adiciona 5 pixels de "borda" preta no meio
            # Cola a imagem original na parte superior
            img_duplicada.paste(img_original, (0, 0))
            # Cola a imagem original novamente na parte inferior
            img_duplicada.paste(img_original, (0, height + 5)) # Adiciona o offset da borda
            
            img_duplicada.save(caminho_saida) # Sobrescreve com a versão duplicada

    except Exception as e:
        print(f"Erro ao processar espectrograma para {os.path.basename(caminho_audio)}: {e}")


# 4. EXECUÇÃO PRINCIPAL 

if __name__ == '__main__':
    if baixar_livro(URL_LIVRO, CAMINHO_LIVRO_TXT):
        frases = carregar_frases(CAMINHO_LIVRO_TXT, QUANTIDADE_DE_FRASES)
        gerar_arquivos_de_audio(frases, PASTA_DE_AUDIOS)

        os.makedirs(PASTA_DE_SAIDA_ESPECTROGRAMAS, exist_ok=True)
        print(f"\n--- Etapa 2: Gerando {VERSOES_POR_AUDIO} versões de espectrograma para cada áudio ---")
        
        lista_de_audios = [f for f in os.listdir(PASTA_DE_AUDIOS) if f.endswith('.mp3')]
        
        for nome_arquivo_audio in tqdm(lista_de_audios, desc="Processando Áudios"):
            caminho_do_audio = os.path.join(PASTA_DE_AUDIOS, nome_arquivo_audio)
            
            for i in range(VERSOES_POR_AUDIO):
                # Randomiza o estilo para cada versão do espectrograma
                estilo_config = {
                    'com_legendas': random.choice([True, False]),
                    'mapa_de_cores': random.choice(LISTA_MAPAS_DE_CORES),
                    'duplicar': random.choice([True, False]), # Aleatoriamente duplica ou não
                    'n_fft': random.choice(N_FFT_OPTIONS),
                    'hop_length': random.choice(HOP_LENGTH_OPTIONS),
                    'n_mels': random.choice(N_MELS_OPTIONS)
                }
                
                # Cria um nome de arquivo único para cada versão com os parâmetros usados
                nome_base = os.path.splitext(nome_arquivo_audio)[0]
                identificador_estilo = (
                    f"{'legenda' if estilo_config['com_legendas'] else 'limpo'}_"
                    f"{'duplo' if estilo_config['duplicar'] else 'simples'}_"
                    f"{estilo_config['mapa_de_cores']}_"
                    f"n{estilo_config['n_fft']}_h{estilo_config['hop_length']}_m{estilo_config['n_mels']}"
                )
                nome_saida = f"{nome_base}_v{i+1}_{identificador_estilo}.png"
                caminho_da_saida = os.path.join(PASTA_DE_SAIDA_ESPECTROGRAMAS, nome_saida)

                gerar_espectrograma_de_audio(caminho_do_audio, caminho_da_saida, estilo_config)
                
        total_final = len(os.listdir(PASTA_DE_SAIDA_ESPECTROGRAMAS))
        print(f"\nPROCESSO COMPLETO! {total_final} espectrogramas super variados foram salvos em '{PASTA_DE_SAIDA_ESPECTROGRAMAS}'")