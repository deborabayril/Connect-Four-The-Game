from __future__ import annotations
from typing import List, Dict, Optional
from collections import Counter, defaultdict
from IPython.display import clear_output
import random
import math
import time
import csv
import os


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
    os.system('cls' if os.name == 'nt' else 'clear') # Terminal
    clear_output(wait = True)  # Jupyter

def print_main_menu(invalid_input):
    clear_terminal()

    print("=========== MENU ===========")
    print("1. Human vs Human")
    print("2. Human vs Computer")
    print("3. Computer vs Computer")
    print("D. Generate small dataset ")
    print("============================\n")

    if (invalid_input):
        print("Invalid input. Please select again.")
    print("Choose an option:", end = " ")

def valid_main_menu_option(chosen_option):
    return chosen_option in ["1", "2", "3", "D"]

def start(chosen_option):
    if (chosen_option == "1"):
        human_vs_human()
    elif (chosen_option == "2"):
        human_vs_computer()
    elif (chosen_option == "3"):
        computer_vs_computer()
    else:
        demonstrate_dataset_generation()

def print_human_vs_computer_menu(invalid_input):
    clear_terminal()

    print("==============================")
    print("1. Monte-Carlo Tree Search")
    print("2. Decision Tree")
    print("==============================\n")

    if (invalid_input):
        print("Invalid input. Please select again.")
    print("Choose an algorithm to play against:", end = " ")

def valid_algorithm(input):
    return input in ["1", "2"]

def print_computer_vs_computer_menu(invalid_input):
    clear_terminal()

    print("==============================")
    print("1. MCTS vs Decision Tree")
    print("2. MCTS vs MCTS")
    print("==============================\n")

    if (invalid_input):
        print("Invalid input. Please select again.")
    print("Choose an algorithm to play against:", end = " ")

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
        print(f"Player {player_color(player)}, choose a column (1-7):", end = " ")
        column = input()

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

def board_to_features(board):
    features = []
    for row in board:
        for cell in row:
            if cell is None:
                features.append(0)
            elif cell == 'X':
                features.append(1)
            else:
                features.append(2)
    return features

def make_move(board, column, player):
    for row in range(5, -1, -1):
        if board[row][column] is None:
            board[row][column] = player
            return True
    return False

def save_turn_data_to_csv(board, player, column, file_name):
    current_board_state = [col for row in board for col in row]
    converted_board_state = [0 if col == None else 1 if col == "X" else 2 for col in current_board_state]
    data = converted_board_state + [1 if player == "X" else 2, column]

    file_is_empty = (not os.path.isfile(file_name)) or os.path.getsize(file_name) == 0

    with open(file_name, mode = "a", newline = "") as file:
        writer = csv.writer(file)

        if file_is_empty:
            header = ""
            header = [f"pos{i}" for i in range(1, len(converted_board_state) + 1)]
            header += ["player", "move"]
            writer.writerow(header)

        writer.writerow(data)

def random_MCTS_generator(limit_of_iterations, limit_of_uct_constant):
    board = create_board()
    root_node = MCTSNode("-", -1, 0, [row[:] for row in board], None)
    iterations = random.randint(1, limit_of_iterations/100) * 100
    uct_constant = random.uniform(0.8, limit_of_uct_constant)
    return MCTS(root_node, iterations, uct_constant)

def generate_dataset(number_of_different_mcts, number_of_games_per_mcts, limit_of_iterations, limit_of_uct_constant, output_dataset_file):

    print(f"\nThe generated dataset will be a set of games of a static MCTS (that will always play first as {player_color('X')}) against random generated MCTSs which will vary in the number of iterations and uct_constant value.")
    print(f"\nOur static MCTS will play with the following characteristics:\n- Iterations: 1000\n- uct_constant: √2 = {math.sqrt(2):.5f}")
    print(f"\nGenerating dataset with {number_of_games_per_mcts} games for each of the {number_of_different_mcts} different MCTSs")

    start_dataset_generation_timer = time.time()

    for i in range(1, number_of_different_mcts + 1): # for each different mcts
        for j in range(1, number_of_games_per_mcts + 1): # make each play number_of_games_per_mcts against each other
            board = create_board()
            mcts_X = MCTS(MCTSNode("-", -1, 0, [row[:] for row in board], None))
            mcts_O = random_MCTS_generator(limit_of_iterations, limit_of_uct_constant)
            current_player = "X"
            turn = 1

            if (j == 1):
                print(Color.BOLD + "\n------------- " + Color.RED + f"Opponent {i}" + Color.RESET + " -------------" + Color.RESET)
                print(f"Opponent characteristics:\n- Iterations: {mcts_O.iterations}\n- uct_constant: {mcts_O.uct_constant:.5f}\n" + Color.RESET)
                print(Color.BOLD + f"Starting game {j}" + Color.RESET)
            else:
                print(Color.BOLD + f"\nStarting game {j}" + Color.RESET)

            start_time = time.time()

            while True:
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
                    mcts_O.root = MCTSNode(mcts_X.root.player, mcts_X.root.move, mcts_X.root.turn,
                                       [row[:] for row in mcts_X.root.board], None)

                save_turn_data_to_csv(board, current_player, column, output_dataset_file)
                player_won, winning_line = check_win(board, current_player)

                if player_won:
                    elapsed_time = time.time() - start_time
                    print(f"Player {player_color(current_player)} won!")
                    print(f"This whole game took {elapsed_time:.3f} seconds.")
                    break

                if check_draw(turn):
                    elapsed_time = time.time() - start_time
                    print("Draw!")
                    print(f"This whole game took {elapsed_time:} seconds.")
                    break

                current_player = oppositePlayer(current_player)
                turn += 1

            if (j == number_of_games_per_mcts):
                print(Color.BOLD + "--------------------------------------" + Color.RESET)

    end_dataset_generation_timer = time.time() - start_dataset_generation_timer
    print(Color.BOLD + Color.GREEN + f"\nFinished dataset generation in {end_dataset_generation_timer:.3f} seconds." + Color.RESET)

def demonstrate_dataset_generation():
    clear_terminal()
    print(Color.BOLD + Color.BLUE + "Examplifying how our dataset was generated through generate_dataset() method with small values for a faster generation." + Color.RESET)
    generate_dataset(3, 2, 1000, 2.4, "test_dataset_generation.csv")

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
    # this method exists to also confirm that the monte-carlo tree was correctly created during the game
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
                mcts = MCTS(MCTSNode(current_player, column, turn, board, None))
                print(Color.BLUE + f"\nMCTS Iterations: {mcts.iterations}\n" + Color.RESET)
            else:
                mcts.update_root(column)

        else:
            print("MCTS thinking...")
            start_time = time.time()
            column = mcts.mcts_move()
            elapsed_time = time.time() - start_time
            last_mcts_move = f"Player {player_color(current_player)} chose column {column + 1} in {elapsed_time:.3f}s"
            make_move(board, column, current_player)
            mcts.update_root(column)

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
    try:
        board = create_board()
        current_player = "X"
        turn = 1
        column = 0
        last_decision_tree_move = ""
        game_history = ""

        decision_tree = create_tree()

        while True:
            print_board(board, turn)

            if current_player == "X":
                print(last_decision_tree_move)
                column = valid_column_value(board, turn, current_player)
                make_move(board, column, current_player)
            else:
                features = board_to_features(board)
                column = decision_tree.predict(features)
                print(f"Decision Tree predicts column: {column + 1}")

                if is_valid_move(board, column):
                    make_move(board, column, current_player)
                    last_decision_tree_move = f"Player {player_color(current_player)} chose column {column + 1}"
                else:
                    valid_moves = [col for col in range(7) if is_valid_move(board, col)]
                    if valid_moves:
                        column = random.choice(valid_moves)
                        make_move(board, column, current_player)
                        last_decision_tree_move = f"Player {player_color(current_player)} chose column {column + 1}"
                    else:
                        break

            player_won, winning_line = check_win(board, current_player)
            game_history += f"Turn {turn}: ({player_color(current_player)}, {column + 1})\n"

            if player_won:
                print_board(board, turn, winning_line)
                print(last_decision_tree_move)
                print(f"Player {player_color(current_player)} won!")
                break

            if check_draw(turn):
                print_board(board, turn)
                print("Draw!")
                break

            current_player = oppositePlayer(current_player)
            turn += 1

        print("\nWould you like to review the game history (y/n)?", end=" ")
        answer = input()
        if answer == "y":
            print(game_history, end="")

    except Exception as e:
        print(f"\nAn error occurred: {e}\n")

def mcts_vs_mcts():
    board = create_board()
    current_player = "X"
    turn = 1
    last_move = ""

    mcts_X = MCTS(MCTSNode("-", -1, 0, [row[:] for row in board], None))
    mcts_O = MCTS(MCTSNode("-", -1, 0, [row[:] for row in board], None))

    while True:
        print_board(board, turn)
        print(last_move)
        print(f"Player {player_color(current_player)} is thinking...")

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
            mcts_O.root = MCTSNode(mcts_X.root.player, mcts_X.root.move, mcts_X.root.turn,
                               [row[:] for row in mcts_X.root.board], None)

        last_move = f"Player {player_color(current_player)} chose column {column + 1}"
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

def mcts_vs_decision_tree():
    board = create_board()
    current_player = "X"
    turn = 1
    last_move = ""
    game_history = ""

    mcts = MCTS(MCTSNode("-", -1, 0, [row[:] for row in board], None))
    decision_tree = create_tree()

    while True:
        print_board(board, turn)
        print(last_move)
        print(f"Player {player_color(current_player)} is thinking...")

        if current_player == "X":
            column = mcts.mcts_move()
            make_move(board, column, current_player)
            mcts.update_root(column)

            if (turn == 1):
                mcts.root.parent = None

        else:
            features = board_to_features(board)
            column = decision_tree.predict(features)

            if is_valid_move(board, column):
                make_move(board, column, current_player)
                last_decision_tree_move = f"Player {player_color(current_player)} chose column {column + 1}"
            else:
                valid_moves = [col for col in range(7) if is_valid_move(board, col)]
                if valid_moves:
                    column = random.choice(valid_moves)
                    make_move(board, column, current_player)
                    last_decision_tree_move = f"Player {player_color(current_player)} chose column {column + 1}"
                else:
                    break

            mcts.update_root(column)

        last_move = f"Player {player_color(current_player)} chose column {column + 1}"
        game_history += f"Turn {turn}: ({player_color(current_player)}, {column + 1})\n"
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

    print("\nWould you like to review the game history (y/n)?", end=" ")
    answer = input()
    if answer == "y":
        print(game_history, end="")


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
        game_history += f"Turn {turn}: ({player_color(current_player)}, {column + 1})\n"

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

    print_human_vs_computer_menu(False)
    algorithm_choice = input()

    while not valid_algorithm(algorithm_choice):
        print_human_vs_computer_menu(True)
        algorithm_choice = input()

    if (algorithm_choice == "1"):
        human_vs_mcts();
    else:
        human_vs_decision_tree();

def computer_vs_computer():

    print_computer_vs_computer_menu(False)
    choice = input()

    while not valid_algorithm(choice):
        print_computer_vs_computer_menu(True)
        choice = input()

    if (choice == "1"):
        mcts_vs_decision_tree()
    else:
        mcts_vs_mcts()


class MCTSNode:
    def __init__(self, player, move, turn, board, parent):
        self.wins = 0
        self.visits = 0
        self.player = player
        self.move = move
        self.turn = turn
        self.board = board
        self.parent = parent
        self.children = []

    def uct(self, uct_constant):
        if self.visits == 0:
            return float("inf")
        return (self.wins / self.visits) + uct_constant * math.sqrt(math.log(self.parent.visits) / self.visits)

    def child_with_move(self, move):
        for child in self.children:
            if child.move == move:
                return child

    def print_all_previous_turns(self):
        lines = []
        node = self
        while node is not None:
            line = f"Turn {node.turn}: ({player_color(node.player)}, {node.move + 1})"
            lines.append(line)
            node = node.parent
        for line in reversed(lines):
            print(line)

class MCTS:
    def __init__(self, root, iterations = 1000, uct_constant = math.sqrt(2)):
        self.root = root
        self.iterations = iterations
        self.uct_constant = uct_constant

    def select_node(self, node):
        best_child = None
        best_uct = -float("inf")

        for child in node.children:
            uct_value = child.uct(self.uct_constant)
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
            child_node = MCTSNode(new_player, move, node.turn + 1, new_board, node)
            node.children.append(child_node)

    def simulate(self, node):
        board = [row[:] for row in node.board]
        current_player = oppositePlayer(node.player)

        while True:
            valid_moves = [col for col in range(7) if is_valid_move(board, col)]
            if not valid_moves:
                return None

            move = random.choice(valid_moves)
            make_move(board, move, current_player)

            if check_win(board, current_player)[0]:
                return current_player

            current_player = oppositePlayer(current_player)

    def backpropagate(self, node, result):
        while node is not None:
            won = 1 if node.player == result else 0
            node.visits += 1
            node.wins += won
            node = node.parent

    def mcts_move(self):
        root = self.root

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
            if child.visits > best_visits:
                best_visits = child.visits
                best_move = child.move

        return best_move

    def update_root(self, chosen_column):
        self.root = self.root.child_with_move(chosen_column)


def load_dataset(path: str) -> (List[List[int]], List[int]):
    X, y = [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # Lê a primeira linha como cabeçalho
        for i, row in enumerate(reader):  # 'row' já é uma lista de strings
            features = [int(val) for val in row[:42]]
            if len(row) > 42:
                turn = int(row[42])
                features.append(turn)
            move = int(row[-1])
            X.append(features)
            y.append(move)
    return X, y

def create_tree():
    X, y = load_dataset("dataset.csv")
    clf = DecisionTree()
    clf.fit(X, y)
    return clf

def entropy(labels: List[int]) -> float:
    total = len(labels)
    counts = Counter(labels)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())

def info_gain(parent_y: List[int], partitions: List[List[int]]) -> float:
    parent_ent = entropy(parent_y)
    total = len(parent_y)
    weighted = sum(len(p) / total * entropy(p) for p in partitions)
    return parent_ent - weighted

class DecisionTreeNode:
    def __init__(self, *, feature: Optional[int] = None, children: Optional[Dict[int, "DecisionTreeNode"]] = None, label: Optional[int] = None):
        self.feature = feature
        self.children = children or {}
        self.label = label

    def is_leaf(self):
        return self.label is not None

class DecisionTree:
    def __init__(self, max_depth: int = 8, min_samples_split: int = 20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: Optional[DecisionTreeNode] = None

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

    def _build(self, X: List[List[int]], y: List[int], depth: int) -> DecisionTreeNode:
        if depth >= self.max_depth or len(set(y)) == 1 or len(X) < self.min_samples_split:
            majority = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(label=majority)
        feat, parts = self._best_split(X, y)
        if feat is None:
            majority = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(label=majority)
        children = {}
        for val, idxs in parts.items():
            sub_X = [X[i] for i in idxs]
            sub_y = [y[i] for i in idxs]
            children[val] = self._build(sub_X, sub_y, depth + 1)
        return DecisionTreeNode(feature=feat, children=children)

    # ----------------------------- previsão --------------------------------
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



def main():
    print_main_menu(False)
    chosen_option = input()

    while not valid_main_menu_option(chosen_option):
        print_main_menu(True)
        chosen_option = input()

    start(chosen_option)

if __name__ == "__main__":
    main()