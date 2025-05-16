import math

def carregar_dados(caminho_arquivo):
    """Carrega os dados do arquivo CSV."""
    dados = []
    with open(caminho_arquivo, 'r') as arquivo:
        linhas = arquivo.readlines()[1:]
        for linha in linhas:
            linha = linha.strip().split(',')
            sepallength = float(linha[1])
            sepalwidth = float(linha[2])
            petallength = float(linha[3])
            petalwidth = float(linha[4])
            classe = linha[5]
            dados.append([sepallength, sepalwidth, petallength, petalwidth, classe])
    return dados

def discretizar_valor(valor, limites):
    """Discretiza um valor contínuo em uma categoria."""
    if valor <= limites[0]:
        return 'baixo'
    elif valor <= limites[1]:
        return 'médio'
    else:
        return 'alto'

def discretizar_dados(dados, limites):
    """Discretiza os atributos do conjunto de dados."""
    dados_discretizados = []
    for amostra in dados:
        sl = discretizar_valor(amostra[0], limites['sepallength'])
        sw = discretizar_valor(amostra[1], limites['sepalwidth'])
        pl = discretizar_valor(amostra[2], limites['petallength'])
        pw = discretizar_valor(amostra[3], limites['petalwidth'])
        classe = amostra[4]
        dados_discretizados.append([sl, sw, pl, pw, classe])
    return dados_discretizados

def calcular_entropia(conjunto_dados):
    """Calcula a entropia de um conjunto de dados."""
    classes = {}
    for amostra in conjunto_dados:
        classe = amostra[-1]
        classes[classe] = classes.get(classe, 0) + 1
    entropia = 0
    total_amostras = len(conjunto_dados)
    for classe in classes:
        probabilidade = classes[classe] / total_amostras
        entropia -= probabilidade * math.log2(probabilidade) if probabilidade > 0 else 0
    return entropia

def calcular_ganho_informacao(conjunto_dados, indice_atributo):
    """Calcula o ganho de informação de um atributo."""
    entropia_inicial = calcular_entropia(conjunto_dados)
    valores_atributo = set(amostra[indice_atributo] for amostra in conjunto_dados)
    entropia_ponderada = 0
    total_amostras = len(conjunto_dados)

    for valor in valores_atributo:
        subconjunto = [amostra for amostra in conjunto_dados if amostra[indice_atributo] == valor]
        probabilidade = len(subconjunto) / total_amostras
        entropia_ponderada += probabilidade * calcular_entropia(subconjunto)

    return entropia_inicial - entropia_ponderada

def encontrar_melhor_atributo(conjunto_dados, atributos):
    """Encontra o melhor atributo para dividir o conjunto de dados."""
    melhor_ganho = -1
    melhor_atributo = None
    melhor_indice = -1
    for indice, atributo in enumerate(atributos):
        ganho = calcular_ganho_informacao(conjunto_dados, indice)
        if ganho > melhor_ganho:
            melhor_ganho = ganho
            melhor_atributo = atributo
            melhor_indice = indice
    return melhor_atributo, melhor_indice

def obter_classe_mais_comum(lista_classes):
    """Retorna a classe mais comum em uma lista."""
    if not lista_classes:
        return None
    counts = {}
    for classe in lista_classes:
        counts[classe] = counts.get(classe, 0) + 1
    return max(counts, key=counts.get)

def construir_arvore_id3(conjunto_dados, atributos, profundidade=0):
    """Constrói recursivamente a árvore de decisão ID3."""
    classes = [amostra[-1] for amostra in conjunto_dados]
    if not conjunto_dados or not atributos:
        return obter_classe_mais_comum(classes)
    if all(c == classes[0] for c in classes):
        return classes[0]

    melhor_atributo, melhor_indice = encontrar_melhor_atributo(conjunto_dados, atributos)

    if melhor_atributo is None:
        return obter_classe_mais_comum(classes)

    arvore = {melhor_atributo: {}}
    valores_atributo = sorted(list(set(amostra[melhor_indice] for amostra in conjunto_dados)))
    atributos_restantes = [attr for attr in atributos if attr != melhor_atributo]

    for valor in valores_atributo:
        subconjunto = [amostra for amostra in conjunto_dados if amostra[melhor_indice] == valor]
        arvore[melhor_atributo][valor] = construir_arvore_id3(subconjunto, atributos_restantes, profundidade + 1)

    return arvore

def classificar_amostra_id3(arvore, amostra_discretizada):
    """Classifica uma amostra discretizada usando a árvore ID3."""
    atributo_raiz = list(arvore.keys())[0]
    ramos = arvore[atributo_raiz]
    indice_atributo = ['sepallength', 'sepalwidth', 'petallength', 'petalwidth'].index(atributo_raiz)
    valor_amostra = amostra_discretizada[indice_atributo]

    if valor_amostra in ramos:
        resultado = ramos[valor_amostra]
        if isinstance(resultado, dict):
            return classificar_amostra_id3(resultado, amostra_discretizada)
        else:
            return resultado
    else:
        # Lidar com valores não vistos durante o treinamento (retorna a classe mais comum do nó pai)
        return obter_classe_mais_comum([folha for lista_folhas in obter_todas_folhas(arvore) for folha in lista_folhas])

def obter_todas_folhas(node):
    """Função auxiliar para obter todas as folhas da árvore."""
    folhas = []
    if isinstance(node, str):
        return [[node]]
    for sub_arvore in node.values():
        folhas.extend(obter_todas_folhas(sub_arvore))
    return folhas

def imprimir_arvore_formatada(arvore, indentacao=""):
    """Imprime a árvore de decisão no formato desejado."""
    if isinstance(arvore, str):
        print(f"{indentacao}'{arvore}'")
        return

    atributo = list(arvore.keys())[0]
    print(f"'{atributo}': {{")
    for valor, sub_arvore in arvore[atributo].items():
        print(f"{indentacao}    '{valor}': ", end='')
        imprimir_arvore_formatada(sub_arvore, indentacao + "    ")
    print(f"{indentacao}}}")

if __name__ == "__main__":
    caminho_arquivo = 'iris.csv'
    dados = carregar_dados(caminho_arquivo)

    # Definir os limites de discretização 
    limites_discretizacao = {
        'sepallength': [5.0, 6.0],
        'sepalwidth': [2.8, 3.3],
        'petallength': [2.0, 4.5],
        'petalwidth': [0.8, 1.6]
    }

    dados_discretizados = discretizar_dados(dados, limites_discretizacao)

    atributos = ['sepallength', 'sepalwidth', 'petallength', 'petalwidth']
    arvore_id3 = construir_arvore_id3(dados_discretizados, atributos)

    print("Árvore de Decisão (ID3):")
    imprimir_arvore_formatada(arvore_id3)

    # Exemplos de classificação (usando os mesmos limites para discretizar)
    exemplos_teste = [
        ([5.5, 3.0, 4.0, 1.3], 'Iris-versicolor'),
        ([6.5, 3.0, 5.5, 2.0], 'Iris-virginica'),
        ([4.8, 3.1, 1.5, 0.2], 'Iris-setosa'),
        ([6.0, 2.2, 5.0, 1.5], 'Iris-virginica'),
        ([5.7, 2.8, 4.1, 1.3], 'Iris-versicolor'),
        ([7.0, 3.2, 4.7, 1.4], 'Iris-versicolor'),
        ([5.0, 3.6, 1.4, 0.2], 'Iris-setosa'),
        ([6.7, 3.0, 5.2, 2.3], 'Iris-virginica'),
        ([5.1, 2.5, 3.0, 1.1], 'Iris-versicolor'),
        ([4.4, 3.0, 1.3, 0.2], 'Iris-setosa')
    ]

    print("\nExemplos de Classificação:")
    for exemplo, real_classe in exemplos_teste:
        sl_d = discretizar_valor(exemplo[0], limites_discretizacao['sepallength'])
        sw_d = discretizar_valor(exemplo[1], limites_discretizacao['sepalwidth'])
        pl_d = discretizar_valor(exemplo[2], limites_discretizacao['petallength'])
        pw_d = discretizar_valor(exemplo[3], limites_discretizacao['petalwidth'])
        amostra_discretizada = [sl_d, sw_d, pl_d, pw_d]
        predicao = classificar_amostra_id3(arvore_id3, amostra_discretizada)
        print(f"Exemplo: ['{sl_d}', '{sw_d}', '{pl_d}', '{pw_d}'], Real: {real_classe}, Previsão: {predicao}")

    # Avaliação de precisão (em todo o conjunto de dados discretizado)
    predicoes_corretas = 0
    for amostra_original in dados:
        sl_d = discretizar_valor(amostra_original[0], limites_discretizacao['sepallength'])
        sw_d = discretizar_valor(amostra_original[1], limites_discretizacao['sepalwidth'])
        pl_d = discretizar_valor(amostra_original[2], limites_discretizacao['petallength'])
        pw_d = discretizar_valor(amostra_original[3], limites_discretizacao['petalwidth'])
        amostra_discretizada = [sl_d, sw_d, pl_d, pw_d]
        real_classe = amostra_original[4]
        predicao = classificar_amostra_id3(arvore_id3, amostra_discretizada)
        if predicao == real_classe:
            predicoes_corretas += 1

    precisao = predicoes_corretas / len(dados)
    print(f"\nResultados do Teste: {predicoes_corretas} corretos de {len(dados)} ({precisao:.2%})")

    if precisao >= 0.9:
        print("\nA árvore de decisão atingiu ou superou a precisão de 90%.")
    else:
        print("A árvore de decisão não atingiu a precisão de 90%.")
        print("Pode ser necessário ajustar os limites de discretização para melhorar a precisão.")
