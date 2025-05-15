import math
from collections import Counter

def entropy(labels):
    total = len(labels)
    counts = Counter(labels)
    return -sum((count/total) * math.log2(count/total) for count in counts.values() if count > 0)

def information_gain(data, labels, feature_index):
    total_entropy = entropy(labels)
    values = set(row[feature_index] for row in data)
    weighted_entropy = 0.0

    for value in values:
        subset_data = [row for row in data if row[feature_index] == value]
        subset_labels = [labels[i] for i in range(len(data)) if data[i][feature_index] == value]
        weight = len(subset_labels) / len(labels)
        weighted_entropy += weight * entropy(subset_labels)

    return total_entropy - weighted_entropy

class Node:
    def __init__(self, feature=None, value=None, children=None, label=None):
        self.feature = feature      # índice da feature usada para dividir
        self.value = value          # valor para comparação (opcional para dados contínuos)
        self.children = children or {}  # dicionário valor -> nó filho
        self.label = label          # só nos nós folha

def id3(data, labels, features):
    if labels.count(labels[0]) == len(labels):
        return Node(label=labels[0])  # Todos da mesma classe

    if not features:
        # Retorna classe mais comum
        most_common = Counter(labels).most_common(1)[0][0]
        return Node(label=most_common)

    # Escolher melhor feature
    gains = [information_gain(data, labels, i) for i in features]
    best_feature = features[gains.index(max(gains))]

    node = Node(feature=best_feature)
    values = set(row[best_feature] for row in data)

    for value in values:
        subset_data = [row for row in data if row[best_feature] == value]
        subset_labels = [labels[i] for i in range(len(data)) if data[i][best_feature] == value]
        if not subset_data:
            most_common = Counter(labels).most_common(1)[0][0]
            node.children[value] = Node(label=most_common)
        else:
            remaining_features = [f for f in features if f != best_feature]
            child = id3(subset_data, subset_labels, remaining_features)
            node.children[value] = child

    return node

def classify(node, example):
    while node.label is None:
        value = example[node.feature]
        if value in node.children:
            node = node.children[value]
        else:
            return None  # ou classe mais comum
    return node.label
