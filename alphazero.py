from typing import List, Tuple
import numpy as np
from mcts import mcts, MctsAction, MctsState
from keras.models import Model


class CatZero():
    def __init__(self, model: Model, iterationLimit=None, timeLimit=None):
        self.__iterationLimit = iterationLimit
        self.__timeLimit = timeLimit
        self.__states: List[Tuple[MctsState, MctsAction]] = []
        self.__model = model

    def play(self, current):
        searcher = mcts(iterationLimit=self.__iterationLimit, timeLimit=self.__timeLimit)

        history = []
        self.__states = []

        while not current.isTerminal():
            current.evaluate(history)

            best_action = searcher.search(current)

            self.__states.append((current, best_action))
            history.append(current)

            print(best_action)

            current = current.takeAction(best_action)

        current.evaluate(history)
        return current

    def learn(self, positions, value: float, probabilities):
        # Translate states to
        #(data: position, label:[reward, played])
        values = np.repeat(value, len(positions))
        positions = np.array(positions)
        probabilities = np.array(probabilities)

        print(positions.shape)
        print(probabilities.shape)

        self.__model.fit(positions, [probabilities, values], epochs=3, batch_size=3)
        print("Finished learning game")

    def get_states(self) -> List[Tuple[MctsState, MctsAction]]:
        return self.__states
