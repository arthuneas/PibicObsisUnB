import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURAÇÃO ---
# Caminho para a pasta com os arquivos de áudio que você baixou
AUDIO_SOURCE_FOLDER = r'C:\Users\arthur.almeida\Downloads\UrbanSound8K\UrbanSound8K\audio\fold1'
# Pasta onde as imagens de espectrogramas serão salvas
SPECTROGRAM_OUTPUT_FOLDER = './dataset/espectrograma'
# --- FIM DA CONFIGURAÇÃO ---


# Função para criar e salvar a imagem do espectrograma
def save_spectrogram(audio_path, save_path):
    try:
        y, sr = librosa.load(audio_path)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        fig, ax = plt.subplots()
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        
        fig.tight_layout(pad=0)
        ax.axis('off')
        
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    except Exception as e:
        print(f"Erro ao processar {audio_path}: {e}")

# Cria a pasta de saída se ela não existir
if not os.path.exists(SPECTROGRAM_OUTPUT_FOLDER):
    os.makedirs(SPECTROGRAM_OUTPUT_FOLDER)

# Itera sobre os arquivos de áudio e gera os espectrogramas
print(f"Procurando arquivos em: {AUDIO_SOURCE_FOLDER}")
for filename in os.listdir(AUDIO_SOURCE_FOLDER):
    if filename.endswith('.wav'):
        audio_path = os.path.join(AUDIO_SOURCE_FOLDER, filename)
        save_path = os.path.join(SPECTROGRAM_OUTPUT_FOLDER, f'{os.path.splitext(filename)[0]}.png')
        save_spectrogram(audio_path, save_path)
        print(f"-> Gerado: {save_path}")

print("Geração de espectrogramas concluída!")