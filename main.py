import random
from connect_four import *

def main():
    board = create_board()
    player = "X"
    iterations = 1000  # Número de iterações do MCTS

    while True:
        print_board(board)

        if player == "X":
            column = int(input(f"Jogador {player}, escolha uma coluna (0-6): "))
        else:
            column = mcts_move(board, player, iterations)
            print(f"Jogador {player} escolheu a coluna {column}")

        if is_valid_move(board, column):
            make_move(board, column, player)

            if check_win(board, player):
                print_board(board)
                print(f"Jogador {player} venceu!")
                break

            if check_draw(board):
                print_board(board)
                print("Empate!")
                break

            player = "O" if player == "X" else "X"
        else:
            print("Movimento inválido. Tente novamente.")

if __name__ == "__main__":
    main()


