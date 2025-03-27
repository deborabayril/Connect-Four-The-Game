from connect_four import *

def main():
    board = create_board()
    player = "X"

    while True:
        print_board(board)
        column = int(input(f"Jogador {player}, escolha uma coluna (0-6): "))

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