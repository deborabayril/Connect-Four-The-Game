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
    iterations = 300  # Número de iterações do MCTS
    str = ""

    # Inicialize a variável 'previous_games' para armazenar o histórico de movimentos
    previous_games = {}

    # Escolha o tipo de personalidade (isso pode ser uma entrada ou uma configuração fixa)
    personality_type = "Agressiva"  # ou "Equilibrado", "Defensiva"

    while True:
        print_board(board)

        if player == "X":
            print(str)
            column = valid_column_value(board, player)
        else:
            column, previous_games = mcts_move(board, player, iterations, personality_type, previous_games)
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
    clear_terminal()
    print("Computador vs Computador iniciado!\n")

    board = create_board()
    player = "X"  # Começa com o jogador X
    iterations = 300  # Número de iterações do MCTS
    personality_type = "Equilibrado"  # Pode ser "Agressiva", "Defensiva" ou "Equilibrado"
    previous_games = {}  # Dicionário vazio para armazenar os jogos anteriores


    while True:
        print_board(board)

        # Jogada do jogador "X"
        if player == "X":
            column, previous_games = mcts_move(board, player, iterations, personality_type, previous_games)
            print(f"Jogador {player} escolheu a coluna {column + 1}")        
        else:
            column, previous_games = mcts_move(board, player, iterations, personality_type, previous_games)
            print(f"Jogador {player} escolheu a coluna {column + 1}")

        make_move(board, column, player)

        print(f"Chamando save_game_to_csv para o jogador {player} na coluna {column + 1}")
        # Salvar o estado do jogo e a jogada no CSV
        save_game_to_csv(board, player, column)

        player_won, winning_line = check_win(board, player)

        if player_won:
            print_board(board, winning_line)
            print(f"Jogador {player} venceu!")
            break

        if check_draw(board):
            print_board(board)
            print("Empate!")
            break

        # Alternar entre X e O
        player = "O" if player == "X" else "X"

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
import random

def simulate(node, exploration_factor=0.1):
    """
    Função de simulação com maior aleatoriedade.
    Explora múltiplas possibilidades durante a simulação.
    """
    board = [row[:] for row in node.board]
    player = node.player

    while True:
        valid_moves = [col for col in range(7) if is_valid_move(board, col)]
        
        if not valid_moves:
            return 0  # Empate (sem movimentos válidos)

        # Aleatoriedade maior na escolha do movimento
        # Exploração pode ser mais ou menos agressiva, dependendo da chance
        if random.random() < exploration_factor:  
            move = random.choice(valid_moves)  # Escolhe um movimento aleatório
        else:
            # Usar heurísticas ou um modelo mais simples para escolher o movimento preferido
            move = max(valid_moves, key=lambda col: evaluate_move(board, col, player))

        make_move(board, move, player)

        # Verificar se alguém venceu
        if check_win(board, player)[0]:
            return 1  # Vitória do jogador atual

        # Alternar jogador
        player = 'O' if player == 'X' else 'X'


def backpropagate(node, result):
    while node is not None:
        node.visits += 1
        node.wins += result
        node = node.parent

def adaptive_strategy(board, player):
    # Verificar se há possibilidade de vitória para o jogador atual ou para o adversário
    player_win, _ = check_win(board, player)
    opponent = 'O' if player == 'X' else 'X'
    opponent_win, _ = check_win(board, opponent)

    if player_win:
        return "Ataque"  # Prioriza vitória
    elif opponent_win:
        return "Defesa"  # Prioriza bloquear o adversário
    else:
        return "Equilibrado"  # Usa MCTS normal


def evaluate_move(board, move, player):
    # Simula a jogada
    new_board = [row[:] for row in board]
    make_move(new_board, move, player)

    # Avaliar a jogada
    points = 0

    # 1 ponto por vitória
    if check_win(new_board, player)[0]:
        points += 1

    # 0.8 ponto por bloqueio
    opponent = 'O' if player == 'X' else 'X'
    if check_win(new_board, opponent)[0]:
        points += 0.8

    # 0.5 ponto por proximidade de vitória
    for row in range(6):
        for col in range(7):
            if new_board[row][col] == player:
                if check_win(new_board, player)[0]:
                    points += 0.5

    # 0.3 ponto por posição estratégica (preferência por centro)
    if move in [3, 4]:  # Posições centrais
        points += 0.3

    return points


def choose_personality(personality_type):
    if personality_type == "Agressiva":
        return "Ataque"
    elif personality_type == "Defensiva":
        return "Defesa"
    else:
        return "Equilibrado"


def learn_from_history(previous_games, current_move, result):
    # Armazenar o movimento e seu resultado no histórico
    previous_games[current_move] = result
    # Este histórico pode ser usado em jogadas futuras
    return previous_games


def mcts_move(board, player, iterations, personality_type, previous_games):
    # Escolhe a estratégia baseada na situação do jogo
    strategy = adaptive_strategy(board, player)

    # Seleciona a personalidade desejada
    personality = choose_personality(personality_type)

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

        # Simulação com base na estratégia
        if strategy == "Ataque":
            # Priorizando a vitória, pontuação de 1
            result = evaluate_move(node.board, node.move, player) + 1
        elif strategy == "Defesa":
            # Priorizando o bloqueio, pontuação de 0.8
            result = evaluate_move(node.board, node.move, player) + 0.8
        else:
            # Estratégia equilibrada, MCTS normal
            result = simulate(node)

        # Retropropagação
        backpropagate(node, result)

    # Escolher o melhor movimento com base na pontuação e histórico
    best_move = None
    best_visits = -1
    best_score = -float('inf')

    for child in root.children:
        score = evaluate_move(board, child.move, player)
        if score > best_score:
            best_score = score
            best_move = child.move
            best_visits = child.visits

    # Aprendizado: Armazena o resultado do movimento e o histórico
    previous_games = learn_from_history(previous_games, best_move, result)

    return best_move, previous_games
