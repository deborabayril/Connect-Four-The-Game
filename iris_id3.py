import csv
import math
import pprint
import random

def carregar_dataset(caminho_arquivo):
    """Carrega o dataset Iris ignorando a coluna ID."""
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            cabecalho = next(reader)
            cabecalho = [col.strip() for col in cabecalho]
            dados = []
            for linha in reader:
                linha = [item.strip() for item in linha]
                if len(linha) == len(cabecalho):
                    # Ignora a primeira coluna (ID)
                    dados.append(linha[1:])
            # Retorna dados e atributos (sem a coluna classe)
            return dados, cabecalho[1:-1]
    except FileNotFoundError:
        print(f"Erro: Arquivo '{caminho_arquivo}' não encontrado.")
        return [], []
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        return [], []

def discretizar_dados(dados):
    """Discretiza os atributos contínuos do dataset Iris."""
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
                linha[4]  # Classe
            ]
            dados_discretizados.append(linha_discretizada)
        except ValueError:
            print(f"Erro ao converter atributos para números na linha: {linha}")
    return dados_discretizados

def dividir_dataset(dados, proporcao_teste=0.2, random_seed=42):
    """Divide o dataset em treino e teste."""
    random.seed(random_seed)
    random.shuffle(dados)
    tamanho_teste = int(len(dados) * proporcao_teste)
    teste_data = dados[:tamanho_teste]
    treino_data = dados[tamanho_teste:]
    X_treino = [linha[:-1] for linha in treino_data]
    y_treino = [linha[-1] for linha in treino_data]
    X_teste = [linha[:-1] for linha in teste_data]
    y_teste = [linha[-1] for linha in teste_data]
    return X_treino, X_teste, y_treino, y_teste

def calcular_entropia(dados):
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
    entropia_inicial = calcular_entropia(dados)
    valores_unicos = set(linha[atributo_index] for linha in dados)
    entropia_dividida = 0
    for valor in valores_unicos:
        dados_filtrados = [linha for linha in dados if linha[atributo_index] == valor]
        probabilidade = len(dados_filtrados) / len(dados)
        entropia_dividida += probabilidade * calcular_entropia(dados_filtrados)
    return entropia_inicial - entropia_dividida

def id3(dados, atributos):
    classes = [linha[-1] for linha in dados]

    # Se todas as classes forem iguais, retorna a classe
    if len(set(classes)) == 1:
        return classes[0]
    # Se não houver atributos, retorna a classe mais comum
    if not atributos:
        return max(set(classes), key=classes.count)

    melhor_atributo = None
    maior_informacao_ganha = -1
    melhor_atributo_index = -1

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
        sub_arvore = id3(
            [linha[:melhor_atributo_index] + linha[melhor_atributo_index+1:] for linha in dados_filtrados],
            atributos_restantes)
        arvore[melhor_atributo][valor] = sub_arvore

    return arvore

def classificar(exemplo, arvore, atributos):
    if not isinstance(arvore, dict):
        return arvore
    atributo = next(iter(arvore))
    if atributo not in atributos:
        return "Desconhecido (Atributo não encontrado)"
    atributo_index = atributos.index(atributo)
    valor = exemplo[atributo_index]
    if valor not in arvore[atributo]:
        return "Desconhecido (Valor não encontrado)"
    return classificar(arvore[atributo][valor], arvore[atributo][valor], atributos) if isinstance(arvore[atributo][valor], dict) else arvore[atributo][valor]

def classificar(exemplo, arvore, atributos):
    if not isinstance(arvore, dict):
        return arvore
    atributo = next(iter(arvore))
    if atributo not in atributos:
        return "Desconhecido (Atributo não encontrado)"
    atributo_index = atributos.index(atributo)
    valor = exemplo[atributo_index]
    if valor not in arvore[atributo]:
        return "Desconhecido (Valor não encontrado)"
    return classificar(exemplo, arvore[atributo][valor], atributos)

def main():
    dados_iris, atributos_originais = carregar_dataset('iris.csv')
    if not dados_iris:
        print("Dataset vazio ou erro ao carregar.")
        return

    dados_discretizados = discretizar_dados(dados_iris)
    atributos_discretizados = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    X_treino, X_teste, y_treino, y_teste = dividir_dataset(dados_discretizados, proporcao_teste=0.2, random_seed=42)

    # Combina atributos e classes para construir a árvore
    dados_treino_completo = [X_treino[i] + [y_treino[i]] for i in range(len(X_treino))]

    arvore = id3(dados_treino_completo, atributos_discretizados)

    print("\nÁrvore de Decisão (ID3):")
    pprint.pprint(arvore)

    corretos = 0
    for i, exemplo in enumerate(X_teste):
        previsao = classificar(exemplo, arvore, atributos_discretizados)
        real = y_teste[i]
        print(f"Exemplo: {exemplo}, Real: {real}, Previsão: {previsao}")
        if previsao == real:
            corretos += 1

    total = len(y_teste)
    precisao = corretos / total if total > 0 else 0
    print(f"\nResultados do Teste: {corretos} corretos de {total} ({precisao:.2%})")

if __name__ == "__main__":
    main()
