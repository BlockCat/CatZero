import network
from enum import Enum
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Convolution2D
from keras.layers import BatchNormalization, Activation, Add, Dense, Flatten
from mcts import MctsAction, MctsState, treeNode


def create_input():
    # One plane for player 1
    # One plane for player 1 previous
    # One plane for player 2
    # One plane for player 2 previous
    # One plane for colour to play
    return Input(shape=(3, 3, 5))


def create_policy(model):
    model = Convolution2D(2, 1, padding="same",
                          data_format="channels_last")(model)
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
    def __init__(self, player: Player, model, probability: float, board=None):
        self.player = player
        self.model = model
        self.probability = probability
        self.reward = 0
        # If no board is provided, create a new board
        if board == None:
            self.board = np.array([[Player.Empty, Player.Empty, Player.Empty, Player.Empty, Player.Empty,
                                    Player.Empty, Player.Empty, Player.Empty, Player.Empty]]).reshape((1, 3, 3))[0]
        else:
            self.board = board

    # When the state is created for the first time, store it and evaluate
    def evaluate(self, node: treeNode):
        if self.isTerminal():

            # check who won
            won_player = None
            if self.checkRow(0) or self.checkCol(0):
                won_player = self.board[0, 0]

            if self.checkRow(1) or self.checkCol(1) or self.checkDiagLeftRight() or self.checkDiagRightLeft():
                won_player = self.board[1, 1]

            if self.checkRow(2) or self.checkCol(2):
                won_player = self.board[2, 2]

            if won_player == None:  # It's a draw :(
                self.reward = 0
            elif won_player == self.player:  # Current player has won
                self.reward = 1
            else:  # Other player has won
                self.reward = -1
        else:
            tensor_input = Exception("Do something")
            # TODO evaluate using model

            # convert board game to nnumpy array

    def getPossibleActions(self):  # Returns an iterable of all actions which can be taken from this state\
        if self.isTerminal():
            return []
        else:
            # TODO
            # Return all empty spaces
            return []

    def takeAction(self, action):  # Returns the state which results from taking action action
        new_board = np.array(self.board, copy=True)
        new_board[action.y, action.x] = action.player
        if action.player == Player.Circle:
            next_player = Player.Cross
        else:
            next_player = Player.Circle
        return TicTacToeState(next_player, self.model, action.probability, new_board)

    def isTerminal(self):  # Returns whether this state is a terminal state
        # Check if all are filled
        if np.all([x != Player.Empty for x in self.board]):
            return True
        # Check three in a row
        if self.checkRow(0) or self.checkRow(1) or self.checkRow(2):
            return True

        # Check column
        if self.checkCol(0) or self.checkCol(1) or self.checkCol(2):
            return True

        # check diagonal
        if self.checkDiagLeftRight() or self.checkDiagRightLeft():
            return True

        return False

    def checkRow(self, row: int) -> bool:
        return self.board[row, 0] == self.board[row, 1] == self.board[row, 2]

    def checkCol(self, col: int) -> bool:
        return self.board[0, col] == self.board[1, col] == self.board[2, col]

    def checkDiagLeftRight(self) -> bool:
        return self.board[0, 0] == self.board[1, 1] == self.board[2, 2]

    def checkDiagRightLeft(self) -> bool:
        return self.board[0, 2] == self.board[1, 1] == self.board[0, 2]

    def getReward(self):  # Returns the reward for this state (predicted using neural network)
        return self.reward

    # Returns the probability of a parent state going into this state (predicted using neural network)
    def getProbability(self):
        return self.probability
