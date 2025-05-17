from __future__ import annotations
import argparse
import csv
import math
import pickle
from collections import Counter, defaultdict
from typing import List, Dict, Optional

def load_dataset(path: str) -> (List[List[int]], List[int]):
    """Lê CSV → (X, y) onde y=move (0‑6)."""
    X, y = [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # Lê a primeira linha como cabeçalho
        print(f"Cabeçalho do CSV: {header}") # Para inspeção
        for i, row in enumerate(reader): # 'row' já é uma lista de strings
            print(f"\n--- Processando linha {i} ---") # Para inspeção
            print(f"row: {row[:5]}...") # Mostra os primeiros 5 valores
            features = [int(val) for val in row[:42]]
            if len(row) > 42:
                turn = int(row[42])
                features.append(turn)
            move = int(row[-1])
            X.append(features)
            y.append(move)
    return X, y

def entropy(labels: List[int]) -> float:
    total = len(labels)
    counts = Counter(labels)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())

def info_gain(parent_y: List[int], partitions: List[List[int]]) -> float:
    parent_ent = entropy(parent_y)
    total = len(parent_y)
    weighted = sum(len(p) / total * entropy(p) for p in partitions)
    return parent_ent - weighted

class Node:
    def __init__(self, *, feature: Optional[int] = None, children: Optional[Dict[int, "Node"]] = None, label: Optional[int] = None):
        self.feature = feature
        self.children = children or {}
        self.label = label
    def is_leaf(self):
        return self.label is not None

class DecisionTreeClassifier:
    def __init__(self, max_depth: int = 8, min_samples_split: int = 20): 
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: Optional[Node] = None

    # ----------------------------- treinamento -----------------------------
    def fit(self, X: List[List[int]], y: List[int]):
        self.root = self._build(X, y, depth=0)

    def _best_split(self, X: List[List[int]], y: List[int]):
        best_gain, best_feat, best_parts = -1, None, None
        n_features = len(X[0])
        for feat in range(n_features):
            parts: Dict[int, List[int]] = defaultdict(list)
            for i, row in enumerate(X):
                parts[row[feat]].append(i)
            splits_y = [[y[i] for i in idxs] for idxs in parts.values()]
            gain = info_gain(y, splits_y)
            if gain > best_gain:
                best_gain, best_feat, best_parts = gain, feat, parts
        return best_feat, best_parts

    def _build(self, X: List[List[int]], y: List[int], depth: int) -> Node:
        if depth >= self.max_depth or len(set(y)) == 1 or len(X) < self.min_samples_split:
            majority = Counter(y).most_common(1)[0][0]
            return Node(label=majority)
        feat, parts = self._best_split(X, y)
        if feat is None:
            majority = Counter(y).most_common(1)[0][0]
            return Node(label=majority)
        children = {}
        for val, idxs in parts.items():
            sub_X = [X[i] for i in idxs]
            sub_y = [y[i] for i in idxs]
            children[val] = self._build(sub_X, sub_y, depth + 1)
        return Node(feature=feat, children=children)

    # ----------------------------- predição --------------------------------
    def predict(self, sample: List[int]) -> int:
        node = self.root
        while node and not node.is_leaf():
            val = sample[node.feature]
            node = node.children.get(val)
            if node is None:
                return 0  # fallback: primeira coluna
        return node.label

    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.root

        indent = "  " * depth
        if node.is_leaf():
            print(f"{indent}Label: {node.label}")
        else:
            print(f"{indent}Feature: col_{node.feature}")
            for value, child in node.children.items():
                print(f"{indent}  Value {value}:")
                self.print_tree(child, depth + 2)

def save_tree(tree: DecisionTreeClassifier, path: str):
    with open(path, "wb") as f:
        pickle.dump(tree, f)

def load_tree(path: str) -> DecisionTreeClassifier:
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    ap = argparse.ArgumentParser(description="Treina árvore ID3 para prever o próximo movimento no Connect‑Four")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--save", default="tree.pkl")
    args = ap.parse_args()
    X, y = load_dataset(args.csv)
    clf = DecisionTreeClassifier(max_depth=args.max_depth)
    clf.fit(X, y)
    save_tree(clf, args.save)
    print(f"Árvore salva em {args.save} (profundidade ≤ {args.max_depth})")

    print("\n--- Testando a árvore carregada ---")

    loaded_tree: DecisionTreeClassifier = load_tree("tree.pkl") # Definição de loaded_tree

    if X:
        print(f"Estrutura de X (primeiros 2 elementos): {X[:2]}")
        print(f"Estrutura de y (primeiros 2 elementos): {y[:2]}")
        if X[0]:
            primeira_amostra = X[0]
            previsao = loaded_tree.predict(primeira_amostra)
            print(f"Para a primeira amostra do dataset:\n{primeira_amostra}")
            print(f"O movimento previsto é: {previsao}, e o movimento real era: {y[0]}")
        else:
            print("A primeira amostra do dataset está vazia.")
    else:
        print("Não há dados para testar a árvore carregada.")

if __name__ == "__main__":
    main()
    # Chame print_tree na instância carregada (loaded_tree)
    loaded_tree = load_tree("tree.pkl")
    print("\n--- Visualização da Árvore ---")
    loaded_tree.print_tree()

#Para chamar a função use python mcts_id3.py --csv dataset.csv --max-depth 8
