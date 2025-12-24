from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

# Ler arquivo WAV
arquivo = r"C:\Users\arthur.almeida\Downloads\clarinet.wav"
fs, data = wavfile.read(arquivo)

# Se for estéreo, pegar apenas um canal
if data.ndim > 1:
    data = data[:, 0]

# Normalizar para float (-1 a 1)
data = data / np.max(np.abs(data))

# Criar espectrograma
plt.figure(figsize=(10, 6))
plt.specgram(data, NFFT=512, Fs=fs, noverlap=256, cmap='viridis')
plt.xlabel("Tempo (s)")
plt.ylabel("Frequência (Hz)")
plt.ylim(100, 5000)  # limitar até 10 kHz
plt.colorbar(label="Intensidade (dB)")

plt.savefig("espectrograma_clarinete.png", dpi=300, bbox_inches='tight')
plt.show()