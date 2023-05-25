import ludopy
import random

class RandomAgent:
    def __init__(self):
        pass
    
    def decide_moves(self, game_state, dice_roll, possible_moves):
        return random.choice(possible_moves)

def main():
    game = ludopy.Game()
    agent = RandomAgent()
    
    while True:
        current_player = game.current_player
        dice = random.randint(1,6)
        
        # Get the moves for the current player
        move_pieces = game.get_move_pieces(dice)
        
        # If there are no valid moves, skip the turn
        if move_pieces:
            if current_player == 0:
                # Human player's turn
                print("Current state:", game.get_board())
                print("Dice roll:", dice)
                print("Possible moves:", move_pieces)
                chosen_move = int(input("Choose a move: "))
                game.move(move_pieces[chosen_move])
            else:
                # Agent's turn
                agent_move = agent.decide_moves(game.get_board(), dice, move_pieces)
                game.move(agent_move)
        else:
            if current_player == 0:
                # Human player's turn
                print("Current state:", game.get_board())
                print("Dice roll:", dice)
                print("No valid moves, skipping turn")
                input("Press enter to continue...")
            else:
                # Agent's turn
                print("Agent has no valid moves, skipping turn")
            
        # Check if the game is over
        if game.get_winner() != -1:
            break
        
    print(f"Player {game.get_winner()} wins!")

main()