import os
import shutil
import random
import math

print("--- Iniciando a divisão do dataset ---")

# --- CONFIGURAÇÃO ---
# Caminho para a pasta que contém as pastas 'espectrograma' e 'nao_espectrograma'
PASTA_ORIGEM = './dataset'

# Caminho para a nova pasta onde o dataset dividido será criado
PASTA_DESTINO = './dataset_final'

# Proporções da divisão (a soma deve ser 1.0)
PROPORCAO_TREINO = 0.7  # 70% para treino
PROPORCAO_VALIDACAO = 0.15 # 15% para validação
PROPORCAO_TESTE = 0.15   # 15% para teste


def dividir_arquivos(nome_classe):
    print(f"\nProcessando classe: {nome_classe}")
    
    # Caminhos completos
    pasta_classe_origem = os.path.join(PASTA_ORIGEM, nome_classe)
    
    # Verifica se a pasta de origem da classe existe
    if not os.path.exists(pasta_classe_origem):
        print(f"!!! ATENÇÃO: Pasta de origem '{pasta_classe_origem}' não encontrada. Pulando esta classe.")
        return

    # Pega a lista de todos os arquivos de imagem na pasta da classe
    arquivos = [f for f in os.listdir(pasta_classe_origem) if os.path.isfile(os.path.join(pasta_classe_origem, f))]
    random.shuffle(arquivos) # Embaralha a lista para garantir aleatoriedade
    
    total_arquivos = len(arquivos)
    print(f"Total de {total_arquivos} imagens encontradas.")

    # Calcula o número de arquivos para cada conjunto
    num_treino = math.ceil(total_arquivos * PROPORCAO_TREINO)
    num_validacao = math.floor(total_arquivos * PROPORCAO_VALIDACAO)
    # O teste fica com o restante para garantir que a soma seja exata
    num_teste = total_arquivos - num_treino - num_validacao

    print(f"Distribuindo: {num_treino} para treino, {num_validacao} para validação, {num_teste} para teste.")

    # Separa as listas de arquivos
    arquivos_treino = arquivos[:num_treino]
    arquivos_validacao = arquivos[num_treino : num_treino + num_validacao]
    arquivos_teste = arquivos[num_treino + num_validacao :]

    # Função auxiliar para criar pastas e copiar arquivos
    def copiar_arquivos(lista_arquivos, conjunto):
        # Cria a pasta de destino se não existir (ex: ./dataset_final/train/espectrograma)
        pasta_destino_conjunto = os.path.join(PASTA_DESTINO, conjunto, nome_classe)
        os.makedirs(pasta_destino_conjunto, exist_ok=True)
        
        # Copia cada arquivo
        for arquivo in lista_arquivos:
            caminho_origem_arquivo = os.path.join(pasta_classe_origem, arquivo)
            caminho_destino_arquivo = os.path.join(pasta_destino_conjunto, arquivo)
            shutil.copyfile(caminho_origem_arquivo, caminho_destino_arquivo)

    # Copia os arquivos para as pastas de destino
    print(f"Copiando arquivos de treino...")
    copiar_arquivos(arquivos_treino, 'train')
    
    print(f"Copiando arquivos de validação...")
    copiar_arquivos(arquivos_validacao, 'validation')
    
    print(f"Copiando arquivos de teste...")
    copiar_arquivos(arquivos_teste, 'test')
    
    print(f"Classe '{nome_classe}' processada com sucesso!")


# EXECUÇÃO
# Lista de classes (nomes das pastas) a serem processadas
classes = ['espectrograma', 'nao_espectrograma']

for nome_da_classe in classes:
    dividir_arquivos(nome_da_classe)

print("\n--- Divisão do dataset concluída com sucesso! ---")
print(f"O dataset final está pronto na pasta: '{PASTA_DESTINO}'")
