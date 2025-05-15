import random
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import math
import time
import csv
import os
from sklearn import datasets


class Color:
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

def player_color(player, won = None):
    if (won):
        return Color.GREEN + Color.BOLD + player + Color.RESET
    elif (player == "X"):
        return Color.YELLOW + Color.BOLD + player + Color.RESET
    else:
        return Color.RED + Color.BOLD + player + Color.RESET

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_main_menu(invalid_input):
    clear_terminal()

    # print("Welcome To Connect Four!\n")
    print("=========== MENU ===========")
    print("1. Human vs Human")
    print("2. Human vs Computer")
    print("3. Computer vs Computer")
    print("============================\n")

    if (invalid_input):
        print("Invalid input. Please select again.")
    print("Choose a game mode:", end = " ")

def valid_game_mode(game_mode):
    return game_mode in ["1", "2", "3"]

def start_game(game_mode):
    if (game_mode == "1"):
        human_vs_human()
    elif (game_mode == "2"):
        human_vs_computer()
    else:
        computer_vs_computer()

def print_algorithm_menu(invalid_input):
    clear_terminal()

    print("==============================")
    print("1. Play against Monte-Carlo")
    print("2. Play against Decision Tree")
    print("==============================\n")

    if (invalid_input):
        print("Invalid input. Please select again.")
    print("Choose an algorithm to play against:", end = " ")

def valid_algorithm(input):
    return input in ["1", "2"]

def create_board():
    return [[None for _ in range(7)] for _ in range(6)]

def print_board(board, turn, winning_positions = None):
    clear_terminal()

    print(f"      Turn {turn}")
    print("1  2  3  4  5  6  7")

    for row in range(6):
        for col in range(7):
            elem = board[row][col]

            if elem is None:
                print(".", end="  ")
            elif winning_positions is not None and (row, col) in winning_positions:
                print(player_color(elem, True), end="  ")
            else:
                print(player_color(elem), end="  ")
        print()
    print()

def is_valid_input(input):
    return input in ["1", "2", "3", "4", "5", "6", "7"]

def is_valid_move(board, column):
    return 0 <= column <= 6 and board[0][column] is None

def valid_column_value(board, turn, player, last_mcts_move = None):

    while True:
        column = input(f"Player {player_color(player)}, choose a column (1-7): ")

        if (is_valid_input(column)):
            column = int(column) - 1
            if (is_valid_move(board, column)):
                return column

        print_board(board, turn)
        if (last_mcts_move is not None):
            print(last_mcts_move)
        print("Invalid move. Try again.")

def oppositePlayer(player):
    if player == "X":
        return "O"
    return "X"

def make_move(board, column, player):
    # verificar baixo para cima a primeira posição na coluna column que está disponível
    for row in range(5, -1, -1):
        if board[row][column] is None:
            board[row][column] = player
            return True
    return False

def save_game_to_csv(board, player, column, file_name = "dataset.csv"):

    current_board_state = [col for row in board for col in row]
    converted_board_state = [0 if col == None else 1 if col == "X" else 2 for col in current_board_state]
    data = converted_board_state + [1 if player == "X" else 2, column]

    # current_board_state_v2 = ["-" if col == None else col for row in board for col in row]
    # data = current_board_state_v2 + [player, column]

    with open(file_name, mode = "a", newline = "") as file:
        writer = csv.writer(file)
        writer.writerow(data)
        # print(f"Salvando dados: {data}")

def check_win(board, player):
    """Verifica se o jogador venceu."""
    # Verificar linhas
    for row in range(6):
        for col in range(4):
            if board[row][col] == board[row][col + 1] == board[row][col + 2] == board[row][col + 3] == player:
                return True, [(row, col), (row, col + 1), (row, col + 2), (row, col + 3)]

    # Verificar colunas
    for col in range(7):
        for row in range(3):
            if board[row][col] == board[row + 1][col] == board[row + 2][col] == board[row + 3][col] == player:
                return True, [(row, col), (row + 1, col), (row + 2, col), (row + 3, col)]

    # Verificar diagonais (da esquerda para a direita)
    for row in range(3):
        for col in range(4):
            if board[row][col] == board[row + 1][col + 1] == board[row + 2][col + 2] == board[row + 3][
                col + 3] == player:
                return True, [(row, col), (row + 1, col + 1), (row + 2, col + 2), (row + 3, col + 3)]

    # Verificar diagonais (da direita para a esquerda)
    for row in range(3):
        for col in range(3, 7):
            if board[row][col] == board[row + 1][col - 1] == board[row + 2][col - 2] == board[row + 3][
                col - 3] == player:
                return True, [(row, col), (row + 1, col - 1), (row + 2, col - 2), (row + 3, col - 3)]

    return False, []

def check_draw(turn):
    return turn == 42

def review_game_history(end_game_node):
    # this method only exists to also confirm that the monte-carlo tree was
    # correctly created during the game
    print("\nWould you like to review the game history (y/n)?", end = " ")
    answer = input()
    if answer == "y":
        end_game_node.print_all_previous_turns()

def human_vs_mcts():
    board = create_board()
    current_player = "X"
    turn = 1
    column = 0
    last_mcts_move = ""

    mcts = None

    while True:
        print_board(board, turn)

        if current_player == "X":
            print(last_mcts_move)
            column = valid_column_value(board, turn, current_player, last_mcts_move)
            make_move(board, column, current_player)
            if (mcts == None):
                mcts = MCTS(MCTS_Node(current_player, column, turn, board, None))
                # print(Color.BLUE + f"\nMCTS Iterations: {mcts.iterations}\n" + Color.RESET)
            else:
                mcts.update_root(column)
            # print(f"\n\nMCTS {mctsChosenColumn + 1} -> Child {column + 1}: {mcts.root.wins} / {mcts.root.visits} = {(mcts.root.wins / (mcts.root.visits)) * 100:.3f}% || uct = {mcts.root.uct():.4f}\n\n")

        else:
            print("MCTS thinking...")
            start_time = time.time()
            column = mcts.mcts_move()
            elapsed_time = time.time() - start_time
            last_mcts_move = f"Player {player_color(current_player)} chose column {column + 1} in {elapsed_time:.3f}s"
            make_move(board, column, current_player)
            mcts.update_root(column)

        save_game_to_csv(board, current_player, column)
        player_won, winning_line = check_win(board, current_player)

        if player_won:
            print_board(board, turn, winning_line)
            print(last_mcts_move)
            print(f"Player {player_color(current_player)} won!")
            break

        if check_draw(turn):
            print_board(board, turn)
            print("Draw!")
            break

        current_player = oppositePlayer(current_player)
        turn += 1

    review_game_history(mcts.root)

def human_vs_decision_tree():
    decision_tree = load_decision_tree_from_dataset()

    board = create_board()
    current_player = "X"
    turn = 1

    while True:
        print_board(board, turn)

        if current_player == "X":
            column = valid_column_value(board, turn, current_player)
            make_move(board, column, current_player)

        else:
            flatten_board_state = [0 if col is None else 1 if col == "X" else 2 for row in board for col in row]
            player_id = 1 if current_player == "X" else 2
            board_input = flatten_board_state + [player_id]

            column = decision_tree.predict([board_input])[0]

            if board[0][column] is not None:
                valid_columns = [c for c in range(len(board[0])) if board[0][c] is None]
                column = random.choice(valid_columns)
                print(f"Predicted column was full. Fallback to column {column + 1}")

            print(f"Decision Tree chose column {column + 1}")
            make_move(board, column, current_player)

        if check_win(board, current_player)[0]:
            print_board(board, turn)
            print(f"Player {current_player} won!")
            break

        if check_draw(turn):
            print("Draw!")
            break

        current_player = oppositePlayer(current_player)
        turn += 1

def mcts_vs_mcts_for_dataset():
    for i in range(1, 31):
        print("\nStarting Game ", i)
        start_time = time.time()
        board = create_board()
        current_player = "X"
        turn = 1
        last_move = ""

        mcts_X = MCTS(MCTS_Node("-", -1, 0, [row[:] for row in board], None))
        mcts_O = MCTS(MCTS_Node("-", -1, 0, [row[:] for row in board], None))
        while True:
            #print_board(board, turn)
            #print(last_move)
            #print(f"Turn {turn}: Player {player_color(current_player)} is thinking...")

            if current_player == "X":
                column = mcts_X.mcts_move()
                make_move(board, column, current_player)
                mcts_X.update_root(column)
                mcts_O.update_root(column)

            else:
                column = mcts_O.mcts_move()
                make_move(board, column, current_player)
                mcts_X.update_root(column)
                mcts_O.update_root(column)

            if turn == 1:
                mcts_X.root.parent = None
                mcts_O.root = MCTS_Node(mcts_X.root.player, mcts_X.root.move, mcts_X.root.turn,
                                   [row[:] for row in mcts_X.root.board], None)

            last_move = f"Player {player_color(current_player)} chose column {column + 1}"
            save_game_to_csv(board, current_player, column)
            player_won, winning_line = check_win(board, current_player)

            if player_won:
                elapsed_time = time.time() - start_time
                # print_board(board, turn, winning_line)
                print(f"Player {player_color(current_player)} won!")
                print(f"This whole game took {elapsed_time:.3f} seconds.")
                break

            if check_draw(turn):
                elapsed_time = time.time() - start_time
                #print_board(board, turn)
                print("Draw!")
                print(f"This whole game took {elapsed_time:.3f} seconds.")
                break

            current_player = oppositePlayer(current_player)
            turn += 1

        #review_game_history(mcts_X.root)

def mcts_vs_mcts():
    board = create_board()
    current_player = "X"
    turn = 1
    last_move = ""

    mcts_X = MCTS(Node("-", -1, 0, [row[:] for row in board], None))
    mcts_O = MCTS(Node("-", -1, 0, [row[:] for row in board], None))

    while True:
        print_board(board, turn)
        print(last_move)
        print(f"Turn {turn}: Player {player_color(current_player)} is thinking...")

        if current_player == "X":
            column = mcts_X.mcts_move()
            make_move(board, column, current_player)
            mcts_X.update_root(column)
            mcts_O.update_root(column)

        else:
            column = mcts_O.mcts_move()
            make_move(board, column, current_player)
            mcts_X.update_root(column)
            mcts_O.update_root(column)

        if turn == 1:
            mcts_X.root.parent = None
            mcts_O.root = Node(mcts_X.root.player, mcts_X.root.move, mcts_X.root.turn,
                               [row[:] for row in mcts_X.root.board], None)

        last_move = f"Player {player_color(current_player)} chose column {column + 1}"
        save_game_to_csv(board, current_player, column)
        player_won, winning_line = check_win(board, current_player)

        if player_won:
            print_board(board, turn, winning_line)
            print(f"Player {player_color(current_player)} won!")
            break

        if check_draw(turn):
            print_board(board, turn)
            print("Draw!")
            break

        current_player = oppositePlayer(current_player)
        turn += 1

    review_game_history(mcts_X.root)

def human_vs_human():
    board = create_board()
    current_player = "X"
    turn = 1
    game_history = ""

    while True:
        print_board(board, turn)
        column = valid_column_value(board, turn, current_player)
        make_move(board, column, current_player)
        player_won, winning_line = check_win(board, current_player)
        game_history += f"Turn {turn}: ({current_player}, {column + 1})\n"

        if player_won:
            print_board(board, turn, winning_line)
            print(f"Player {player_color(current_player)} chose column {column + 1}")
            print(f"Player {player_color(current_player)} won!")
            break

        if check_draw(turn):
            print_board(board, turn)
            print("Draw!")
            break

        current_player = oppositePlayer(current_player)
        turn += 1

    print("\nWould you like to review the game history (y/n)?", end = " ")
    answer = input()
    if answer == "y":
        print(game_history, end = "")

def human_vs_computer():

    print_algorithm_menu(False)
    algorithm_choice = input()

    while not valid_algorithm(algorithm_choice):
        print_algorithm_menu(True)
        algorithm_choice = input()

    if (algorithm_choice == "1"):
        human_vs_mcts();
    else:
        human_vs_decision_tree();

def computer_vs_computer():
    #mcts_vs_mcts()
    mcts_vs_mcts_for_dataset()

class MCTS_Node:
    def __init__(self, player, move, turn, board, parent):
        self.wins = 0
        self.visits = 0
        self.player = player
        self.move = move
        self.turn = turn
        self.board = board
        self.parent = parent
        self.children = []

    def uct(self):
        if self.visits == 0:
            return float("inf")
        return (self.wins / self.visits) + math.sqrt(2) * math.sqrt(math.log(self.parent.visits) / self.visits)

    def child_with_move(self, move):
        for child in self.children:
            if child.move == move:
                return child

    def print_all_previous_turns(self):
        lines = []
        node = self
        while node is not None:
            line = f"Turn {node.turn}: ({node.player}, {node.move + 1})"
            lines.append(line)
            node = node.parent
        for line in reversed(lines):
            print(line)

class MCTS:
    def __init__(self, root, iterations = 1000):
        self.root = root
        self.iterations = iterations

    def select_node(self, node):
        best_child = None
        best_uct = -float("inf")

        for child in node.children:
            uct_value = child.uct()
            if uct_value > best_uct:
                best_uct = uct_value
                best_child = child

        return best_child

    def expand_node(self, node):
        valid_moves = [col for col in range(7) if is_valid_move(node.board, col)]
        for move in valid_moves:
            new_board = [row[:] for row in node.board]
            new_player = oppositePlayer(node.player)
            make_move(new_board, move, new_player)
            child_node = MCTS_Node(new_player, move, node.turn + 1, new_board, node)
            node.children.append(child_node)

    def simulate(self, node):
        board = [row[:] for row in node.board]
        current_player = oppositePlayer(node.player)
        #print(f"\n\nInitiating Simulation. Last Player: {node.player} Column: {node.move} Turn: {node.turn}")
        #node.print_all_previous_turns()
        #print_board(board, node.turn)
        while True:
            valid_moves = [col for col in range(7) if is_valid_move(board, col)]
            if not valid_moves:
                return None

            move = random.choice(valid_moves)
            make_move(board, move, current_player)
            #player_won, winning_line = check_win(board, current_player)
            #print("\nSimulating ...")
            #print_board(board, 0, winning_line)

            if check_win(board, current_player)[0]:
                return current_player  # Vitória do jogador atual

            current_player = oppositePlayer(current_player)

    def backpropagate(self, node, result):
        while node is not None:
            won = 1 if node.player == result else 0
            node.visits += 1
            node.wins += won
            node = node.parent

    def mcts_move(self):
        root = self.root
        # print(f"\n\n------------------ Starting Root ------------------\nWins: {root.wins}\nVisits: {root.visits}\nTurn: {root.turn}\nPlayer: {root.player}\nMove: {root.move + 1}\nNumber Children: {len(root.children)}\nBoard:")
        # print_board(root.board, root.turn)

        for _ in range(self.iterations):
            node = root

            while node.children:
                node = self.select_node(node)

            if not check_win(node.board, oppositePlayer(node.player))[0] and not check_draw(node.turn):
                self.expand_node(node)
                if node.children:
                    node = random.choice(node.children)

            result = self.simulate(node)
            self.backpropagate(node, result)

        best_move = None
        best_visits = -1

        for child in root.children:
            # print(f"Child {child.move + 1}: {child.wins} / {child.visits} = {(child.wins / (child.visits)) * 100:.3f}% || uct = {child.uct():.4f}")
            if child.visits > best_visits:
                best_visits = child.visits
                best_move = child.move

        # print(f"\n------------------ Updated Root  ------------------\nWins: {root.wins}\nVisits: {root.visits}\nTurn: {root.turn}\nPlayer: {root.player}\nMove: {root.move + 1}\nNumber Children: {len(root.children)}\nBoard:")
        return best_move

    def update_root(self, chosen_column):
        self.root = self.root.child_with_move(chosen_column)

class DecisionTree_Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # check the stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return DecisionTree_Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # find the best split
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            leaf_value = self._most_common_label(y)
            return DecisionTree_Node(value=leaf_value)

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return DecisionTree_Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                # calculate the information gain
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        # parent entropy
        parent_entropy = self._entropy(y)

        # create children
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # calculate the weighted avg. entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # calculate the IG
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

def load_decision_tree_from_dataset():
    df = pd.read_csv("dataset.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # If y contains strings like "3", convert to int
    # y = y.astype(int)

    clf = DecisionTree(max_depth=20)
    clf.fit(X, y)
    return clf

def testing_decision_tree_with_iris():
    df = pd.read_csv("iris.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Encode string labels to integers
    # le = LabelEncoder()
    # y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.0, random_state=1234
    )

    clf = DecisionTree(max_depth = 20)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    def accuracy(y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)

    acc = accuracy(y_test, predictions)
    print(f"Accuracy {acc:.4f}")