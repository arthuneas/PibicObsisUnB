
<div align="center">
  <h1> PIBIC: Deep Learning para Classificação Sismológica</h1>
  <p><i>Implementação de Redes Neurais Convolucionais para Classificação de Eventos Sísmicos</i></p>
</div>

<br>

<div align="center">
  <img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Architecture-Deep_Neural_Networks-F7DF1E?style=for-the-badge&logo=pytorch&logoColor=black" alt="Redes Neurais">
  <img src="https://img.shields.io/badge/Signal_Processing-DSP-007ACC?style=for-the-badge&logo=python&logoColor=white" alt="DSP">
  <br>
  <img src="https://img.shields.io/badge/UnB-005128?style=for-the-badge" alt="UnB">
  <img src="https://img.shields.io/badge/Status-Finalizado-green?style=for-the-badge" alt="Status">
</div>

<br>

<hr>

##  Resumo da Pesquisa
Este repositório apresenta o ecossistema de treinamento e avaliação de modelos de Deep Learning desenvolvidos no Observatório Sismológico da Universidade de Brasília (Obsis/UnB).
A pesquisa investiga a eficácia de Redes Neurais Convolucionais (CNNs) na classificação automatizada de sinais sismográficos, comparando duas frentes de processamento: o uso de sinais puros (séries temporais brutas) e a análise visual de espectrogramas.
O projeto foi executado de forma gradual, progredindo do básico ao avançado — desde as arquiteturas de redes mais simples até os modelos de estado da arte. Além das implementações práticas, o repositório detalha a fundamentação teórica de cada etapa realizada, garantindo o rigor científico em todo o processo de desenvolvimento.


##  Metodologia de Classificação

O modelo foi treinado para distinguir entre diferentes categorias de sinais (ex: sismos naturais, explosões e ruído ambiental) utilizando duas frentes:

1.  **Sinais Puros:** Classificação direta de vetores de 5000 amostras.
2.  **Espectrogramas:** Análise de imagens 224x224 representando a densidade espectral de potência do sinal no tempo.


##  Tecnologias de Destaque
- Deep Learning: PyTorch (Redes Neurais 1D e 2D).
- Processamento de Sinais: ObsPy, SciPy e NumPy.
- Visualização: Matplotlib e Librosa (Geração de Espectrogramas).


##  Resultados e Avaliação
O sistema gera automaticamente relatórios detalhados de classificação e matrizes de confusão para validar a precisão dos modelos em dados reais do Observatório Sismológico, garantindo a reprodutibilidade científica dos experimentos.

<br>

<hr>

<p align="center">
  Pesquisa desenvolvida por <strong>Arthur Santos e Andreza Rodrigues</strong><br>
  Orientado por <strong>Dr. Ricardo Lopes de Queiroz</strong>
</p>
