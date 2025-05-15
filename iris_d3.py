import csv
import math
import pprint
from sklearn.model_selection import train_test_split

__all__ = ["id3", "classificar"]

def carregar_dataset(caminho_arquivo):
    """Carrega o dataset Iris a partir de um arquivo CSV."""
    try:
        with open(caminho_arquivo, 'r') as file:
            reader = csv.reader(file)
            cabecalho = next(reader)  # Ler o cabeçalho
            dados = [linha for linha in reader]
            return dados, cabecalho[:-1] # Retorna os dados e os nomes dos atributos
    except FileNotFoundError:
        print(f"Erro: Arquivo '{caminho_arquivo}' não encontrado.")
        return [], []
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        return [], []

def discretizar_dados(dados):
    """Discretiza os atributos contínuos do dataset."""
    dados_discretizados = []
    for linha in dados:
        try:
            sepal_length = float(linha[0])
            sepal_width = float(linha[1])
            petal_length = float(linha[2])
            petal_width = float(linha[3])

            linha_discretizada = [
                'baixo' if sepal_length < 5.0 else 'médio' if sepal_length < 6.5 else 'alto',
                'baixo' if sepal_width < 2.8 else 'médio' if sepal_width < 3.2 else 'alto',
                'baixo' if petal_length < 2.0 else 'médio' if petal_length < 4.0 else 'alto',
                'baixo' if petal_width < 0.2 else 'médio' if petal_width < 0.6 else 'alto',
                linha[4] # Classe
            ]
            dados_discretizados.append(linha_discretizada)
        except ValueError:
            print(f"Erro ao converter atributos para números na linha: {linha}")
    return dados_discretizados

def calcular_entropia(dados):
    """Calcula a entropia de um conjunto de dados."""
    total = len(dados)
    contagem_classes = {}
    for linha in dados:
        classe = linha[-1]
        contagem_classes[classe] = contagem_classes.get(classe, 0) + 1

    entropia = 0
    for count in contagem_classes.values():
        probabilidade = count / total
        entropia -= probabilidade * math.log2(probabilidade)
    return entropia

def calcular_informacao_ganha(dados, atributo_index):
    """Calcula o ganho de informação de um atributo."""
    entropia_inicial = calcular_entropia(dados)
    valores_unicos = set(linha[atributo_index] for linha in dados)
    entropia_dividida = 0
    for valor in valores_unicos:
        dados_filtrados = [linha for linha in dados if linha[atributo_index] == valor]
        probabilidade = len(dados_filtrados) / len(dados)
        entropia_dividida += probabilidade * calcular_entropia(dados_filtrados)
    return entropia_inicial - entropia_dividida

def id3(dados, atributos):
    """Implementação do algoritmo ID3 para construir uma árvore de decisão."""
    classes = [linha[-1] for linha in dados]

    if len(set(classes)) == 1:
        return classes[0]
    if not atributos:
        return max(set(classes), key=classes.count)

    melhor_atributo = None
    maior_informacao_ganha = -1

    for i, atributo in enumerate(atributos):
        informacao_ganha = calcular_informacao_ganha(dados, i)
        if informacao_ganha > maior_informacao_ganha:
            maior_informacao_ganha = informacao_ganha
            melhor_atributo = atributo
            melhor_atributo_index = i

    arvore = {melhor_atributo: {}}
    valores_unicos = set(linha[melhor_atributo_index] for linha in dados)

    atributos_restantes = [a for a in atributos if a != melhor_atributo]

    for valor in valores_unicos:
        dados_filtrados = [linha for linha in dados if linha[melhor_atributo_index] == valor]
        sub_arvore = id3(dados_filtrados, atributos_restantes)
        arvore[melhor_atributo][valor] = sub_arvore

    return arvore

def classificar(exemplo, arvore, atributos):
    """Classifica um exemplo usando a árvore de decisão."""
    if isinstance(arvore, dict):
        atributo = list(arvore.keys())[0]
        if atributo not in atributos:
            return "Desconhecido (Atributo não encontrado)"
        atributo_index = atributos.index(atributo)
        valor = exemplo[atributo_index]
        if valor not in arvore[atributo]:
            return "Desconhecido (Valor não encontrado)"
        return classificar(exemplo, arvore[atributo][valor], atributos)
    else:
        return arvore

def main():
    """Função principal para carregar, treinar e avaliar o modelo ID3."""
    dados_iris, atributos_originais = carregar_dataset('iris.csv')

    if dados_iris:
        dados_discretizados = discretizar_dados(dados_iris)

        # Separar features (X) e target (y)
        X = [linha[:-1] for linha in dados_discretizados]
        y = [linha[-1] for linha in dados_discretizados]

        # Dividir os dados em treinamento e teste
        X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

        # Treinar a árvore ID3
        arvore = id3([X_treino[i] + [y_treino[i]] for i in range(len(X_treino))], atributos_originais)

        print("\nÁrvore de Decisão (ID3):")
        pprint.pprint(arvore)

        # Avaliar o modelo
        corretos = 0
        for i, exemplo in enumerate(X_teste):
            previsao = classificar(exemplo, arvore, atributos_originais)
            real = y_teste[i]
            print(f"Exemplo: {exemplo}, Real: {real}, Previsão: {previsao}")
            if previsao == real:
                corretos += 1

        total = len(y_teste)
        precisao = corretos / total
        print(f"\nResultados do Teste: {corretos} corretos de {total} ({precisao:.2%})")

if __name__ == "__main__":
    main()

