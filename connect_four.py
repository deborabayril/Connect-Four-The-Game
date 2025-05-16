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

def save_turn_data_to_csv(board, player, column, file_name):
    current_board_state = [col for row in board for col in row]
    converted_board_state = [0 if col == None else 1 if col == "X" else 2 for col in current_board_state]
    data = converted_board_state + [1 if player == "X" else 2, column]

    with open(file_name, mode = "a", newline = "") as file:
        writer = csv.writer(file)
        writer.writerow(data)

def random_MCTS_generator(limit_of_iterations, limit_of_uct_constant):
    board = create_board()
    root_node = MCTS_Node("-", -1, 0, [row[:] for row in board], None)
    iterations = random.randint(1, limit_of_iterations/100) * 100
    uct_constant = random.uniform(0.8, limit_of_uct_constant)
    return MCTS(root_node, iterations, uct_constant)

def generate_dataset(number_of_different_mcts, number_of_games_per_mcts, limit_of_iterations, limit_of_uct_constant, output_fataset_file):

    print(f"\nThe generated dataset will be a set of games of a static MCTS (that will always play first as {player_color('X')}) against random generated MCTSs which will vary in the number of iterations and uct_constant value.")
    print(f"\nOur static MCTS will play with the following characteristics:\n- Iterations: 1000\n- uct_constant: √2 = {math.sqrt(2):.5f}")
    print(f"\nGenerating dataset with {number_of_games_per_mcts} games for each of the {number_of_different_mcts} different MCTSs")

    start_dataset_generation_timer = time.time()

    for i in range(1, number_of_different_mcts + 1): # for each different mcts
        for j in range(1, number_of_games_per_mcts + 1): # make each play number_of_games_per_mcts against each other
            board = create_board()
            mcts_X = MCTS(MCTS_Node("-", -1, 0, [row[:] for row in board], None))
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
                    mcts_O.root = MCTS_Node(mcts_X.root.player, mcts_X.root.move, mcts_X.root.turn,
                                       [row[:] for row in mcts_X.root.board], None)

                save_turn_data_to_csv(board, current_player, column, output_fataset_file)
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
    print(Color.BOLD + Color.BLUE + "Examplifying how the dataset is generated through generate_dataset() method with small values for a faster generation." + Color.RESET)
    generate_dataset(4, 2, 1000, 2.4, "test_dataset_demonstration.csv")

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
                print(Color.BLUE + f"\nMCTS Iterations: {mcts.iterations}\n" + Color.RESET)
            else:
                mcts.update_root(column)
            # print(f"\n\nMCTS {mctsChosenColumn + 1} -> Child {column + 1}: {mcts.root.wins} / {mcts.root.visits} = {(mcts.root.wins / (mcts.root.visits)) * 100:.3f}% || uct = {mcts.root.uct(self.uct_constant):.4f}\n\n")

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
    clear_terminal()
    print("\nDecision Tree is under development!\n")

def mcts_vs_mcts():
    board = create_board()
    current_player = "X"
    turn = 1
    last_move = ""

    mcts_X = MCTS(MCTS_Node("-", -1, 0, [row[:] for row in board], None))
    mcts_O = MCTS(MCTS_Node("-", -1, 0, [row[:] for row in board], None))

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
            mcts_O.root = MCTS_Node(mcts_X.root.player, mcts_X.root.move, mcts_X.root.turn,
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
    mcts_vs_mcts()

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
            line = f"Turn {node.turn}: ({node.player}, {node.move + 1})"
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
            # print(f"Child {child.move + 1}: {child.wins} / {child.visits} = {(child.wins / (child.visits)) * 100:.3f}% || uct = {child.uct(self.uct_constant):.4f}")
            if child.visits > best_visits:
                best_visits = child.visits
                best_move = child.move

        # print(f"\n------------------ Updated Root  ------------------\nWins: {root.wins}\nVisits: {root.visits}\nTurn: {root.turn}\nPlayer: {root.player}\nMove: {root.move + 1}\nNumber Children: {len(root.children)}\nBoard:")
        return best_move

    def update_root(self, chosen_column):
        self.root = self.root.child_with_move(chosen_column)