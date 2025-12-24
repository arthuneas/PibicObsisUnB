import os
import shutil
import random
from tqdm import tqdm 

# CONFIGURAÇÃO
PASTA_ORIGEM = './dataset_instrumentos_otimizado'
PASTA_DESTINO = './dataset_instrumentos_final'
PROPORCOES = {'train': 0.7, 'validation': 0.15, 'test': 0.15}

def dividir_arquivos_de_uma_classe(nome_classe, pasta_origem, pasta_destino, proporcoes):
    """
    Pega todos os arquivos de uma única classe, embaralha e copia para as
    pastas de treino, validação e teste de destino.
    """
    caminho_completo_origem = os.path.join(pasta_origem, nome_classe)
    
    arquivos = [f for f in os.listdir(caminho_completo_origem) if f.endswith('.png')]
    random.shuffle(arquivos)
    
    total_arquivos = len(arquivos)
    ponto_corte_treino = int(total_arquivos * proporcoes['train'])
    ponto_corte_valid = ponto_corte_treino + int(total_arquivos * proporcoes['validation'])

    conjuntos = {
        'train': arquivos[:ponto_corte_treino],
        'validation': arquivos[ponto_corte_treino:ponto_corte_valid],
        'test': arquivos[ponto_corte_valid:]
    }
    
    for nome_conjunto, lista_arquivos in conjuntos.items():
        pasta_destino_final = os.path.join(pasta_destino, nome_conjunto, nome_classe)
        os.makedirs(pasta_destino_final, exist_ok=True)
        
        for arquivo in lista_arquivos:
            shutil.copy(
                os.path.join(caminho_completo_origem, arquivo),
                pasta_destino_final
            )

def main():

    # Função principal que orquestra todo o processo de divisão do dataset.
    print("--- Iniciando a divisão do dataset ---")
    
    if os.path.exists(PASTA_DESTINO):
        shutil.rmtree(PASTA_DESTINO)
        print(f"Pasta de destino antiga '{PASTA_DESTINO}' removida.")
    
    try:
        classes = [d for d in os.listdir(PASTA_ORIGEM) if os.path.isdir(os.path.join(PASTA_ORIGEM, d))]
        if not classes:
            print(f"ERRO: Nenhuma pasta de classe encontrada em '{PASTA_ORIGEM}'.")
            return
        print(f"Classes detectadas: {', '.join(classes)}")
    except FileNotFoundError:
        print(f"ERRO: Pasta de origem '{PASTA_ORIGEM}' não encontrada.")
        return

    for nome_classe in tqdm(classes, desc="Processando Classes"):
        dividir_arquivos_de_uma_classe(nome_classe, PASTA_ORIGEM, PASTA_DESTINO, PROPORCOES)

    print(f"\n--- Divisão concluída com sucesso! ---")
    print(f"Dataset final pronto em: '{PASTA_DESTINO}'")

if __name__ == '__main__':
    main()