# The plan:
# Execute MCTS in rust
# game logic in rust
# network evaluation in python

# Start python
# Load network
# Start first game
# retrieve game moves from rust
# select using neuralnetwork
# retrieve mcts in rust
# Do best move (and store state and move selected)
# once end go back and learn
import numpy as np
import random
from tictactoe import Player, TicTacToeState, TicTacToeAction, create_model
from mcts import mcts

print("Hello world")

# XO_
# _X_
# __Ox



# One plane for player 1
# One plane for player 1 previous
# One plane for player 2
# One plane for player 2 previous
# One plane for colour to play
#
# state = np.array([[
#     [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
#     [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
#     [[0, 1, 0], [0, 0, 0], [0, 0, 1]],
#     [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
#     [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
# ]])
#
# print(state.shape)

# result = model.predict(state)
#
# print("Policy")
# print(np.reshape(result[0], (3, 3)))
#
#
# print("Value")
# print(result[1][0,0])

def test1(model):

    current = TicTacToeState(Player.Cross, model, 0)
    prev = []
    while not current.isTerminal():
        current.pretty_print()
        current.evaluate(prev)
        actions = current.getPossibleActions()
        chosen = random.choice(actions)
        print(current.reward)
        for action in actions:
            print(action)
        print(chosen)
        prev = [current]
        current = current.takeAction(chosen)

    current.pretty_print()
    current.evaluate(prev)
    print(current.getReward())
    print(current.getPossibleActions())

def test2(model):
    current = TicTacToeState(Player.Cross, model, 0)
    prev = []

    searcher = mcts(iterationLimit=100)

    while not current.isTerminal():
        current.pretty_print()
        current.evaluate(prev)

        best_action = searcher.search(current)
        print(best_action)
        prev = [current]
        current = current.takeAction(best_action)

    current.pretty_print()
    current.evaluate(prev)
    print(current.getWinner())
    if current.getWinner() is Player.Cross:
        print("Reward: 1")
    elif current.getWinner() is Player.Circle:
        print("Reward: -1")
    else:
        print("Reward: 0")



model = create_model()
#test1(model)
test2(model)