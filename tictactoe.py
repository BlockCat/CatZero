import network
from enum import Enum
import numpy as np

from typing import List
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Convolution2D
from keras.layers import BatchNormalization, Activation, Add, Dense, Flatten
from keras.regularizers import l2
from mcts import MctsAction, MctsState


def create_input():
    # One plane for player 1
    # One plane for player 2
    # One plane for player 1 previous
    # One plane for player 2 previous
    # One plane for colour to play
    return Input(batch_shape=(None, 5, 3, 3))


def create_policy(model, reg_constant):
    model = Convolution2D(2, 1, padding="same",
                          data_format="channels_first", kernel_regularizer=l2(reg_constant))(model)
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
    def __init__(self, x: int, y: int, probability: float):
        self.x = x
        self.y = y
        self.probability = probability

    def __str__(self):
        return "TicTacToeAction [x:{}, y:{}, p:{}]".format(self.x, self.y, self.probability)

# Represents the game and game logic, oh boi i miss rust already why did python not have traits/interfaces again?
class TicTacToeState(MctsState):
    def __init__(self, player: Player, model, probability: float, board=None):
        self.player = player
        self.model = model
        self.probability = probability
        self.reward = 0
        self.actions = []
        self.action_probs = []
        self.winner = None
        # If no board is provided, create a new board
        if board is None:
            self.board = np.array([[Player.Empty, Player.Empty, Player.Empty, Player.Empty, Player.Empty,
                                    Player.Empty, Player.Empty, Player.Empty, Player.Empty]]).reshape((1, 3, 3))[0]
        else:
            self.board = board

    # When the state is created for the first time, store it and evaluate
    def evaluate(self, prev: List['TicTacToeState']):
        if self.isTerminal():

            # check who won
            self.winner = None
            if self.checkRow(0) or self.checkCol(0):
                self.winner = self.board[0, 0]

            if self.checkRow(1) or self.checkCol(1) or self.checkDiagLeftRight() or self.checkDiagRightLeft():
                self.winner = self.board[1, 1]

            if self.checkRow(2) or self.checkCol(2):
                self.winner = self.board[2, 2]

            if self.winner is None:  # It's a draw :(
                self.reward = 0
            elif self.winner == self.player:  # Current player has won
                self.reward = 1
            else:  # Other player has won
                self.reward = -1
        else:

            input = self.get_neural_input(prev)
            result = self.model.predict(np.array([input]))
            self.action_probs = result[0].reshape((3,3))
            action_probs = self.action_probs + np.random.dirichlet(np.repeat(0.3, 3), 3)

            # print(self.action_probs)

            for y in range(3):
                for x in range(3):
                    if self.board[y, x] is Player.Empty:
                        self.actions.append(TicTacToeAction(x, y, action_probs[y, x]))

            self.reward = result[1][0, 0]

    def get_action_probs(self):
        return self.action_probs.flatten()

    def get_possible_actions(self) -> List[TicTacToeAction]:  # Returns an iterable of all actions which can be taken from this state\
        if self.isTerminal():
            return []
        else:
            return self.actions

    def takeAction(self, action: TicTacToeAction):  # Returns the state which results from taking action action
        new_board = np.array(self.board, copy=True)
        new_board[action.y, action.x] = self.player
        if self.player == Player.Circle:
            next_player = Player.Cross
        else:
            next_player = Player.Circle
        return TicTacToeState(next_player, self.model, action.probability, new_board)

    def isTerminal(self):  # Returns whether this state is a terminal state
        # Check if all are filled
        if np.all([x is not Player.Empty for x in self.board.flatten()]):
            self.winner = Player.Empty
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
        return self.board[row, 0] == self.board[row, 1] == self.board[row, 2] and self.board[row, 0] is not Player.Empty

    def checkCol(self, col: int) -> bool:
        return self.board[0, col] == self.board[1, col] == self.board[2, col] and self.board[0, col] is not Player.Empty

    def checkDiagLeftRight(self) -> bool:
        return self.board[0, 0] == self.board[1, 1] == self.board[2, 2] and self.board[1, 1] is not Player.Empty

    def checkDiagRightLeft(self) -> bool:
        return self.board[0, 2] == self.board[1, 1] == self.board[2, 0] and self.board[1, 1] is not Player.Empty

    def get_reward(self):  # Returns the reward for this state (predicted using neural network)
        return self.reward

    # Returns the probability of a parent state going into this state (predicted using neural network)
    def getProbability(self):
        return self.probability

    def get_winner(self) -> Player:
        return self.winner

    def to_numpy_array(self):
        # TODO: Flatten and then reshape? Could probably be done better
        cross = np.fromiter((1 if x is Player.Cross else 0 for x in self.board.flatten()), dtype=np.int8).reshape(
            (3, 3))
        circle = np.fromiter((1 if x is Player.Circle else 0 for x in self.board.flatten()), dtype=np.int8).reshape(
            (3, 3))
        return [cross, circle]

    def get_neural_input(self, prev: List['TicTacToeState']):
        colour_plane = np.repeat(0 if self.player is Player.Cross else 1, 9).reshape((3, 3))
        current_planes = self.to_numpy_array()

        if prev is None or len(prev) == 0:
            prev_planes = [np.repeat(0, 9).reshape((3, 3)), np.repeat(0, 9).reshape((3, 3))]
        else:
            prev_planes = prev[-1].to_numpy_array()

        return np.array([current_planes[0], current_planes[1], prev_planes[0], prev_planes[1], colour_plane])

    def pretty_print(self):
        print("┌───┐")
        for y in range(3):
            f = "│"
            for x in range(3):
                cell = self.board[y, x]
                if cell == Player.Cross:
                    f += 'X'
                elif cell == Player.Circle:
                    f += 'O'
                else:
                    f += ' '
            print(f + '│')
        print("└───┘")
