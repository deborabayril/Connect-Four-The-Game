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