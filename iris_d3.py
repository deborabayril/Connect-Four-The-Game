import csv
import math
import pprint
__all__ = ["id3", "classificar"]

# Função para carregar o dataset iris a partir de um arquivo CSV
def carregar_dataset(caminho_arquivo):
    try:
        with open(caminho_arquivo, 'r') as file:
            reader = csv.reader(file)
            dados = [linha for linha in reader]
            dados = dados[1:]  # Remover o cabeçalho
            return dados
    except FileNotFoundError:
        print(f"Erro: Arquivo '{caminho_arquivo}' não encontrado.")
        return []
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        return []

# Função para discretizar os dados (transformar atributos contínuos em discretos)
def discretizar_dados(dados):
    for linha in dados:
        try:
            # Discretizando o comprimento da pétala
            petal_length = float(linha[2])  # Atributo do comprimento da pétala
            linha[2] = 'baixo' if petal_length < 2.0 else 'médio' if petal_length < 4.0 else 'alto'

            # Discretizando a largura da pétala
            petal_width = float(linha[3])  # Atributo da largura da pétala
            linha[3] = 'baixo' if petal_width < 0.2 else 'médio' if petal_width < 0.6 else 'alto'

            # Discretizando o comprimento da sépala
            sepal_length = float(linha[0])
            linha[0] = 'baixo' if sepal_length < 5.0 else 'médio' if sepal_length < 6.5 else 'alto'

            # Discretizando a largura da sépala
            sepal_width = float(linha[1])
            linha[1] = 'baixo' if sepal_width < 2.8 else 'médio' if sepal_width < 3.2 else 'alto'
        
        except ValueError:
            print(f"Erro ao converter atributos para números na linha: {linha}")
    return dados

# Função para calcular a entropia dos dados
def calcular_entropia(dados):
    total = len(dados)
    contagem_classes = {}
    
    # Contagem de ocorrências de cada classe (última coluna)
    for linha in dados:
        classe = linha[-1]
        contagem_classes[classe] = contagem_classes.get(classe, 0) + 1
    
    # Calcular a entropia
    entropia = 0
    for classe in contagem_classes.values():
        probabilidade = classe / total
        entropia -= probabilidade * math.log2(probabilidade)
    
    return entropia

# Função para calcular a informação ganha de um atributo
def calcular_informacao_ganha(dados, atributo_index):
    entropia_inicial = calcular_entropia(dados)
    valores_unicos = set(linha[atributo_index] for linha in dados)
    
    # Calcular a entropia após divisão por atributo
    entropia_dividida = 0
    for valor in valores_unicos:
        dados_filtrados = [linha for linha in dados if linha[atributo_index] == valor]
        probabilidade = len(dados_filtrados) / len(dados)
        entropia_dividida += probabilidade * calcular_entropia(dados_filtrados)
    
    return entropia_inicial - entropia_dividida

# Função recursiva para implementar o algoritmo ID3
def id3(dados, atributos):
    classes = [linha[-1] for linha in dados]
    
    # Caso base 1: Se todos os dados têm a mesma classe
    if len(set(classes)) == 1:
        return classes[0]
    
    # Caso base 2: Se não há mais atributos para dividir
    if not atributos:
        return max(set(classes), key=classes.count)
    
    # Calcular o atributo com maior ganho de informação
    melhor_atributo, maior_informacao_ganha = None, -1
    for i, atributo in enumerate(atributos):
        informacao_ganha = calcular_informacao_ganha(dados, i)
        if informacao_ganha > maior_informacao_ganha:
            maior_informacao_ganha, melhor_atributo = informacao_ganha, i
    
    # Construir o nó da árvore de decisão com o melhor atributo
    arvore = {atributos[melhor_atributo]: {}}
    valores_unicos = set(linha[melhor_atributo] for linha in dados)
    
    # Dividir os dados com base nos valores do melhor atributo
    for valor in valores_unicos:
        dados_filtrados = [linha for linha in dados if linha[melhor_atributo] == valor]
        sub_arvore = id3(dados_filtrados, [a for i, a in enumerate(atributos) if i != melhor_atributo])
        arvore[atributos[melhor_atributo]][valor] = sub_arvore
    
    return arvore

def classificar(exemplo, arvore, atributos):
    if isinstance(arvore, dict):
        atributo = list(arvore.keys())[0]
        atributo_index = atributos.index(atributo)
        valor = exemplo[atributo_index]

        if valor not in arvore[atributo]:
            return "Desconhecido"  # ou retorna a classe mais comum, se preferires

        return classificar(exemplo, arvore[atributo][valor], atributos)
    else:
        return arvore

# Função principal para testar o algoritmo
def main():
    dados_iris = carregar_dataset('iris.csv')
    
    if dados_iris:
        dados_discretizados = discretizar_dados(dados_iris)
        atributos = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        
        arvore = id3(dados_discretizados, atributos)

        # Imprimir a árvore de decisão de forma legível
        print("\nÁrvore de Decisão (ID3):")
        pprint.pprint(arvore)
        
        # Testando a classificação de um exemplo
        for exemplo in dados_discretizados:
            previsao = classificar(exemplo, arvore, atributos)
            print(f"Exemplo: {exemplo}, Previsão: {previsao}")

if __name__ == "__main__":
    main()
