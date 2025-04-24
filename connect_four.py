import math
import random
import os
import csv

class color:
    RED = '\033[91m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_menu(invalid_input):
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
    return game_mode in ["1","2","3"]

def start_game(game_mode):
    if (game_mode == 1):
        human_vs_human()
    elif (game_mode == 2):
        human_vs_computer()
    else:
        computer_vs_computer()

def player_color(player, won = None):
    if (won):
        return color.GREEN + color.BOLD + player + color.RESET
    elif (player == "X"):
        return color.YELLOW + color.BOLD + player + color.RESET
    else:
        return color.RED + color.BOLD + player + color.RESET

def is_valid_input(input):
    return input in ["1", "2", "3", "4", "5", "6", "7"]

def valid_column_value(board, player):
    column = input(f"Jogador {player_color(player)}, escolha uma coluna (1-7): ")

    while not is_valid_input(column):
        print_board(board)
        print("Movimento inválido. Tente novamente.")
        column = input(f"Jogador {player_color(player)}, escolha uma coluna: ")
        if (is_valid_input(column)):
            if is_valid_move(board, int(column) - 1):
                break

    return int(column) - 1

def save_game_to_csv(board, player, column, file_name="jogos_connect_four.csv"):
    # Registra o estado do jogo antes da jogada
    state = str(board)
    
    # Dados a serem salvos
    data = [state, player, column + 1]  # O +1 é para converter a coluna para a contagem humana (1-7)
    
    # Salva no arquivo CSV
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)
        print(f"Salvando dados: {data}")

def human_vs_human():

    board = create_board()
    player = "X"

    while True:
        print_board(board)
        column = valid_column_value(board, player)
        make_move(board, column, player)

        player_won, winning_line = check_win(board, player)

        if player_won:
            print_board(board, winning_line)
            print(f"Jogador {player_color(player)} venceu!")
            break

        if check_draw(board):
            print_board(board)
            print("Empate!")
            break

        player = "O" if player == "X" else "X"

def human_vs_computer():
    board = create_board()
    player = "X"
    iterations = 1000  # Número de iterações do MCTS
    str = ""

    while True:
        print_board(board)

        if player == "X":
            print(str)
            column = valid_column_value(board, player)
            save_game_to_csv(board, player, column)  # Salvando o estado após jogada humana
        else:
            column = mcts_move(board, player, iterations)
            save_game_to_csv(board, player, column)  # Salvando o estado após jogada do computador
            str = f"Jogador {player_color(player)} escolheu a coluna {column + 1}"

        make_move(board, column, player)

        player_won, winning_line = check_win(board, player)

        if player_won:
            print_board(board, winning_line)
            print(f"Jogador {player_color(player)} venceu!")
            break

        if check_draw(board):
            print_board(board)
            print("Empate!")
            break

        player = "O" if player == "X" else "X"

def computer_vs_computer():
    board = create_board()
    player = "X"
    iterations = 1000  # Número de iterações do MCTS
    round_counter = 1

    while True:
        print_board(board)
        print(f"Rodada {round_counter}: Jogador {player_color(player)} está pensando...")

        column = mcts_move(board, player, iterations)
        save_game_to_csv(board, player, column)  # Salvando o estado após jogada da IA
        make_move(board, column, player)

        print(f"Jogador {player_color(player)} escolheu a coluna {column + 1}")
        
        player_won, winning_line = check_win(board, player)

        if player_won:
            print_board(board, winning_line)
            print(f"Jogador {player_color(player)} venceu!")
            break

        if check_draw(board):
            print_board(board)
            print("Empate!")
            break

        player = "O" if player == "X" else "X"
        round_counter += 1

def create_board():
    return [[None for _ in range(7)] for _ in range(6)]

def print_board(board, winning_positions = None):
    clear_terminal()

    print("1  2  3  4  5  6  7")

    for row in range(6):
        for col in range(7):
            elem = board[row][col]

            if elem is None:
                print(".", end = "  ")
            elif winning_positions is not None and (row, col) in winning_positions:
                print(player_color(elem, True), end = "  ")
            else:
                print(player_color(elem), end = "  ")
        print()
    print()

def is_valid_move(board, column):
    # somente temos  que verificar se a linha do topo ainda está vazia (None)
    return 0 <= column <= 6 and board[0][column] is None

def make_move(board, column, player):
    # verificar baixo para cima a primeira posição na coluna column que está disponível
    for row in range(5, -1, -1):
        if board[row][column] is None:
            board[row][column] = player
            return True
    return False

def check_win(board, player):
    """Verifica se o jogador venceu."""
    # Verificar linhas
    for row in range(6):
        for col in range(4):
            if board[row][col] == board[row][col+1] == board[row][col+2] == board[row][col+3] == player:
                return True, [(row, col), (row, col+1), (row, col+2), (row, col+3)]

    # Verificar colunas
    for col in range(7):
        for row in range(3):
            if board[row][col] == board[row+1][col] == board[row+2][col] == board[row+3][col] == player:
                return True, [(row, col), (row+1, col), (row+2, col), (row+3, col)]

    # Verificar diagonais (da esquerda para a direita)
    for row in range(3):
        for col in range(4):
            if board[row][col] == board[row+1][col+1] == board[row+2][col+2] == board[row+3][col+3] == player:
                return True, [(row, col), (row+1, col+1), (row+2, col+2), (row+3, col+3)]

    # Verificar diagonais (da direita para a esquerda)
    for row in range(3):
        for col in range(3, 7):
            if board[row][col] == board[row+1][col-1] == board[row+2][col-2] == board[row+3][col-3] == player:
                return True, [(row, col), (row+1, col-1), (row+2, col-2), (row+3, col-3)]

    return False, []

def check_draw(board):
    """Verifica se o jogo terminou em empate."""
    for row in board:
        for cell in row:
            if cell is None:
                return False  # Ainda há espaços vazios
    return True  # Grade cheia

class Node:
    def __init__(self, board, player, move=None, parent=None):
        self.board = board
        self.player = player
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0

    def uct(self, total_visits):
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + 2 * math.sqrt(math.log(total_visits) / self.visits)

def select_node(node):
    total_visits = sum(child.visits for child in node.children)
    best_child = None
    best_uct = -float('inf')

    for child in node.children:
        uct_value = child.uct(total_visits)
        if uct_value > best_uct:
            best_uct = uct_value
            best_child = child

    return best_child

def expand_node(node):
    valid_moves = [col for col in range(7) if is_valid_move(node.board, col)]
    for move in valid_moves:
        new_board = [row[:] for row in node.board]
        make_move(new_board, move, node.player)
        new_player = 'O' if node.player == 'X' else 'X'
        child_node = Node(new_board, new_player, move, node)
        node.children.append(child_node)

def simulate(node):
    board = [row[:] for row in node.board]
    player = node.player

    while True:
        valid_moves = [col for col in range(7) if is_valid_move(board, col)]
        if not valid_moves:
            return 0  # Empate

        move = random.choice(valid_moves)
        make_move(board, move, player)

        if check_win(board, player)[0]:
            return 1  # Vitória do jogador atual

        player = 'O' if player == 'X' else 'X'

def backpropagate(node, result):
    while node is not None:
        node.visits += 1
        node.wins += result
        node = node.parent

def mcts_move(board, player, iterations):
    root = Node(board, player)

    for _ in range(iterations):
        node = root
        # Seleção
        while node.children:
            node = select_node(node)

        # Expansão
        if not check_win(node.board, 'X' if player == 'O' else 'O')[0] and not check_draw(node.board):
            expand_node(node)
            if node.children:
                node = random.choice(node.children)

        # Simulação
        result = simulate(node)

        # Retropropagação
        backpropagate(node, result)

    # Escolher o melhor movimento
    best_move = None
    best_visits = -1

    for child in root.children:
        if child.visits > best_visits:
            best_visits = child.visits
            best_move = child.move

    return best_move
