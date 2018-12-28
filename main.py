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
import tictactoe

import mcts

print("Hello world")

model = tictactoe.create_model()

state = np.array([[
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
]])

probs = np.array([
    [1, 0, 1], [0, 0.5, 0], [1, 0, 1]
])

value = np.array([[0]])

state = state.reshape((1, 3, 3, 5))
probs = probs.reshape(((1, 9)))
value = value.reshape((1, 3, 3, 1))

result = model.predict(state)

#model.fit(state, [probs, value], epochs = 2)



print("Policy")
print(result[0])

print("Value")
print(result[1])