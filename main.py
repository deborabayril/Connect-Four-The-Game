from connect_four import *

def main():
    print_menu(False)
    game_mode = input()

    while game_mode not in ["1","2","3"]:
        print_menu(True)
        game_mode = input()

    start_game(int(game_mode))

if __name__ == "__main__":
    main()
