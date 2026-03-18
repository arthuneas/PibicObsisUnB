import matplotlib.pyplot as plt
import numpy as np
import os
import random

# --- CONFIGURAÇÃO ---
OUTPUT_PLOTS_FOLDER = './dataset/diversos_plots/plots'
NUM_PLOTS_TO_GENERATE = 5000 # Gere quantos gráficos quiser

# Cria o diretório de saída se ele não existir
if not os.path.exists(OUTPUT_PLOTS_FOLDER):
    os.makedirs(OUTPUT_PLOTS_FOLDER)

print(f"Gerando {NUM_PLOTS_TO_GENERATE} gráficos em '{OUTPUT_PLOTS_FOLDER}'...")

# Listas de opções para maior aleatoriedade
plot_types = ['line', 'scatter', 'bar', 'hist', 'pie', 'boxplot']
linestyles = ['-', '--', '-.', ':']
markers = ['.', ',', 'o', 'v', '^', '<', '>', 's', 'p', '*', 'h', '+', 'x', 'D']
words = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta', 'Value', 'Index', 'Time', 'Measurement']

def generate_random_text():
    """Gera um texto simples para títulos e rótulos."""
    return ' '.join(random.sample(words, 2))

for i in range(NUM_PLOTS_TO_GENERATE):
    fig, ax = plt.subplots(figsize=(4, 3)) # Define um tamanho para consistência
    
    # Escolhe um tipo de gráfico aleatoriamente
    plot_type = np.random.choice(plot_types)
    
    # --- GERAÇÃO DOS GRÁFICOS ---
    if plot_type == 'line':
        num_lines = np.random.randint(1, 4) # 1 a 3 linhas no gráfico
        for _ in range(num_lines):
            ax.plot(np.random.rand(10) * 10, 
                    color=np.random.rand(3,), 
                    linestyle=np.random.choice(linestyles), 
                    linewidth=np.random.uniform(1, 3),
                    marker=np.random.choice(markers),
                    markersize=np.random.uniform(3, 8))

    elif plot_type == 'scatter':
        ax.scatter(np.random.rand(20) * 10, 
                   np.random.rand(20) * 10,
                   s=np.random.rand(20) * 100, # Tamanho dos pontos
                   c=np.random.rand(20, 3),     # Cores dos pontos
                   alpha=np.random.uniform(0.5, 1.0))

    elif plot_type == 'bar':
        num_bars = np.random.randint(3, 10)
        ax.bar(range(num_bars), 
               np.random.rand(num_bars) * 10, 
               color=[np.random.rand(3,) for _ in range(num_bars)], # Cor por barra
               alpha=np.random.uniform(0.6, 1.0))

    elif plot_type == 'hist':
        # Gera dados de uma distribuição normal para um histograma mais "real"
        data = np.random.randn(np.random.randint(50, 200)) * 2 + 5
        ax.hist(data, 
                bins=np.random.randint(5, 20), 
                color=np.random.rand(3,), 
                alpha=np.random.uniform(0.6, 1.0))
    
    elif plot_type == 'pie':
        # Gráfico de pizza ignora a maioria das customizações de eixos
        sizes = np.random.rand(np.random.randint(3, 6))
        ax.pie(sizes, 
               colors=[np.random.rand(3,) for _ in range(len(sizes))],
               autopct='%1.1f%%' if np.random.choice([True, False]) else None)
        ax.axis('equal') # Garante que a pizza seja um círculo

    elif plot_type == 'boxplot':
        # Gera múltiplos conjuntos de dados para o boxplot
        num_boxes = np.random.randint(2, 6)
        data = [np.random.randn(50) * np.random.uniform(0.5, 2) + np.random.uniform(-2, 2) for _ in range(num_boxes)]
        ax.boxplot(data)
        
    # --- CUSTOMIZAÇÃO ALEATÓRIA DE EIXOS E ESTILOS ---
    show_axes = np.random.choice([True, False], p=[0.8, 0.2]) # 80% de chance de mostrar eixos
    
    if show_axes and plot_type != 'pie':
        ax.set_title(generate_random_text())
        ax.set_xlabel(generate_random_text())
        ax.set_ylabel(generate_random_text())
        
        if np.random.choice([True, False]):
             ax.grid(True, linestyle='--', alpha=0.6)
    else:
        ax.axis('off')

    fig.tight_layout(pad=0.1)
    
    # --- SALVAR O ARQUIVO ---
    filename = os.path.join(OUTPUT_PLOTS_FOLDER, f'plot_{i}_{plot_type}.png')
    fig.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close(fig) # Fecha a figura para liberar memória

print(f"\n{NUM_PLOTS_TO_GENERATE} gráficos diversos gerados com sucesso!")
