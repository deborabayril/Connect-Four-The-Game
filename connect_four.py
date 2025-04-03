import math
import random
import os

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

def print_player(player):
    # outra possibilidade de print "●"

    if (player == "X"):
        return color.YELLOW + color.BOLD + player + color.RESET
    else:
        return color.RED + color.BOLD + player + color.RESET

def human_vs_human():

    board = create_board()
    player = "X"

    while True:
        print_board(board)
        column = int(input(f"Jogador {print_player(player)}, escolha uma coluna (1-7): "))
        column -= 1

        while not is_valid_move(board, column):
            print_board(board)
            print("Movimento inválido. Tente novamente.")
            column = int(input(f"Jogador {print_player(player)}, escolha uma coluna: "))
            column -= 1

        make_move(board, column, player)

        if check_win(board, player):
            print_board(board)
            print(f"Jogador {print_player(player)} venceu!")
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
            column = int(input(f"Jogador {print_player(player)}, escolha uma coluna (1-7): "))
            column -= 1
        else:
            column = mcts_move(board, player, iterations)
            #print(f"Jogador {player} escolheu a coluna {column + 1}")
            str = f"Jogador {print_player(player)} escolheu a coluna {column + 1}"


        if is_valid_move(board, column):
            make_move(board, column, player)

            if check_win(board, player):
                print_board(board)
                print(f"Jogador {print_player(player)} venceu!")
                break

            if check_draw(board):
                print_board(board)
                print("Empate!")
                break

            player = "O" if player == "X" else "X"
        else:
            print("Movimento inválido. Tente novamente.")

def computer_vs_computer():
    clear_terminal()
    print("Under Development!")

def create_board():
    return [[None for _ in range(7)] for _ in range(6)]

def print_board(board):
    clear_terminal()

    print("1  2  3  4  5  6  7")
    for row in board:
        for elem in row:
            if (elem == None):
                print(".", end = "  ")
            elif (elem == "X"):
                print(print_player(elem), end = "  ")
            else:
                print(print_player(elem), end = "  ")
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
    for row in board:
        for col in range(4):
            if row[col] == row[col+1] == row[col+2] == row[col+3] == player:
                return True

    # Verificar colunas
    for col in range(7):
        for row in range(3):
            if board[row][col] == board[row+1][col] == board[row+2][col] == board[row+3][col] == player:
                return True

    # Verificar diagonais (da esquerda para a direita)
    for row in range(3):
        for col in range(4):
            if board[row][col] == board[row+1][col+1] == board[row+2][col+2] == board[row+3][col+3] == player:
                return True

    # Verificar diagonais (da direita para a esquerda)
    for row in range(3):
        for col in range(3, 7):
            if board[row][col] == board[row+1][col-1] == board[row+2][col-2] == board[row+3][col-3] == player:
                return True

    return False

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

        if check_win(board, player):
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
        if not check_win(node.board, 'X' if player == 'O' else 'O') and not check_draw(node.board):
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