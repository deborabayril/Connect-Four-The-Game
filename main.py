from connect_four import *

def main():
    print_menu(False)
    game_mode = input()

    while not valid_game_mode(game_mode):
        print_menu(True)
        game_mode = input()

    start_game(game_mode)

if __name__ == "__main__":
    main()
