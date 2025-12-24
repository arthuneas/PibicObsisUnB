import h5py
import numpy as np

# Substitua pelo caminho real do seu arquivo
file_path = r'C:\Users\arthur.almeida\Downloads\dados_sismicos.h5'

def carregar_dados(path):
    with h5py.File(path, 'r') as f:
        # Listar as chaves para confirmar os nomes (ex: 'X', 'y', 'classes')
        print("Variáveis no arquivo:", list(f.keys()))
        
        # Carregar para arrays numpy
        # Assumindo que os nomes internos sejam 'X', 'y' e 'classes'
        # Se forem diferentes, ajuste as strings abaixo
        X = f['X'][:] 
        y = f['y'][:]
        
        # Tratamento para as strings das classes (geralmente vêm codificadas em bytes no h5)
        classes_raw = f['classes'][:] # Nome hipotético da 3ª variável
        classes = [c.decode('utf-8') if isinstance(c, bytes) else c for c in classes_raw]
        
    return X, y, classes

# Uso:
X, y, classes = carregar_dados(file_path)
print(f"Shape de X: {X.shape}") # Deve ser (14733, 5000)
print(f"Classes: {classes}")