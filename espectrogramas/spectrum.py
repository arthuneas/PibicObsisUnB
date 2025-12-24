import matplotlib.pyplot as plt 
import numpy as np

#taxa de amostrgem em hz
fs = 1000
#periodo de 0 a 2 segundos
t = np.linspace(0, 2, 2 * fs)
#x é a juncao das senoides (sinais), sendo a funcão np.sin(parametros da função)
x = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)

plt.figure(figsize=(8, 4))
           
#Nesse método seguinte, temos parametros fundamentais para a criaçâo do espectrograma           
#O primeiro parametro (x) define a funcao/sinal trabalhada

#O segundo, NFFT define a quantidade de pontos processados por vez. 
#Maior NFFT, garantimos uma melhor resolução em frequência, pior resolução em tempo.
#Menor NFFT, garantimos uma melhor resolução em tempo, pior em frequência.

#O fs é a taxa de amostragem define a quantidade de amotras por segundo o sinal possui

#cmap é a coloração do espectrograma
plt.specgram(x, NFFT = 256, Fs = fs, cmap = 'viridis')
plt.xlabel("tempo (s)")
plt.ylabel("Frequência (Hz)")
plt.colorbar(label = "intensidade (dB)")
#caso eu queira estabelecer o foco em uma faixa especifica:
#plt.ylim(0, 200), aqui limitamos até 200hz

plt.savefig("espectrograma.png", dpi=300, bbox_inches='tight')
plt.show()