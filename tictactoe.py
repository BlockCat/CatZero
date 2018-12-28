import network
from enum import Enum
import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Convolution2D
from keras.layers import BatchNormalization, Activation, Add, Dense, Flatten
from mcts import MctsAction, MctsState


def create_input():
    # One plane for player 1
    # One plane for player 1 previous
    # One plane for player 2
    # One plane for player 2 previous
    # One plane for colour to play
    return Input(shape=(3, 3, 5))


def create_policy(model):
    model = Convolution2D(2, 1, padding="same", data_format="channels_last")(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Flatten()(model)
    model = Dense(9)(model)
    return Activation("softmax", name="policy_h")(model)


def create_model() -> Model:
    return network.create_model(create_input(), create_policy)


def evaluate(model: Model, state):
    prediction = model.predict(state)
    print(prediction)

class Player(Enum):
    Empty = 0
    Cross = 1
    Circle = 2

class TicTacToeAction(MctsAction):
    def __init__(self, x: int, y: int, player: Player, probability: float):
        self.x = x
        self.y = y
        self.player = player
        self.probability = probability

# Represents the game and game logic, oh boi i miss rust already why did python not have traits/interfaces again?
class TicTacToeState(MctsState):
    def __init__(self, model: Model, probability: float, board = None):
        self.model = model
        self.probability = probability
        self.reward = 0

        # If no board is provided, create a new board
        if board == None:
            self.board = np.array([[Player.Empty, Player.Empty, Player.Empty, Player.Empty, Player.Empty, Player.Empty, Player.Empty, Player.Empty, Player.Empty]]).reshape((3, 3))[0]        
        else:
            self.board = board

    # When the state is created for the first time, store it and evaluate
    def evaluate(self):
        if self.isTerminal():
            # TODO
            # Set the reward to the appropriate value
            # to who won
        else:
            tensor_input = Exception("Do something")
            # TODO evaluate using model



    def getPossibleActions(self): # Returns an iterable of all actions which can be taken from this state\
        if self.isTerminal():
            return []
        else:
            # TODO
            # Return all empty spaces


    def takeAction(self, action): # Returns the state which results from taking action action
        new_board = np.array(self.board, copy=True)
        new_board[action.y][action.x] = action.player
        return TicTacToeState(model, action.probability, new_board)


    def isTerminal(self): # Returns whether this state is a terminal state
        # TODO
        # Count if all are filled
        # Check three in a row        


    def getReward(self): # Returns the reward for this state (predicted using neural network)
        # TODO

    def getProbability(self): # Returns the probability of a parent state going into this state (predicted using neural network)    
        # TODO
