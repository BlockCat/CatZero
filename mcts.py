# Implementation copied from
# https://github.com/pbsinclair42/MCTS
#

import time
import math
import random
from abc import ABC
from typing import List, Type


# In our case,
# we traverse the existing tree until leaf is found
# leaf node is expanded and that state is evaluated by the neural network (P(s, .), V(s)) = nn(s)
# where P(s, .) are predicted good move probabilities (stored in outgoing edges)
# where V(s) is predicted game outcome (win, lose, draw)
# back propagate to top


# state needs to implement:
# evaluate(node): When the state is created for the first time, store it and evaluate
# getPossibleActions(): Returns an iterable of all actions which can be taken from this state
# takeAction(action): Returns the state which results from taking action action
# isTerminal(): Returns whether this state is a terminal state
# getReward(): Returns the reward for this state (predicted using neural network)
# getProbability(): Returns the probability of a parent state going into this state (predicted using neural network)

class treeNode():
    def __init__(self, state, parent):
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0        
        self.children = {}

class MctsAction(ABC):
    pass

class MctsState(ABC):
    def evaluate(self, prev: List['MctsState']): # When the state is created for the first time, store it and evaluate
        pass
    def getPossibleActions(self) -> List[MctsAction]: # Returns an iterable of all actions which can be taken from this state
        pass
    def takeAction(self, action: List[MctsAction]) -> 'MctsState': # Returns the state which results from taking action action
        pass
    def isTerminal(self) -> bool: # Returns whether this state is a terminal state
        pass
    def getReward(self) -> float: # Returns the reward for this state (predicted using neural network)
        pass
    def getProbability(self) -> float: # Returns the probability of a parent state going into this state (predicted using neural network)
        pass


class mcts():
    def __init__(self, timeLimit=None, iterationLimit=None, explorationConstant=1 / math.sqrt(2)):
        if timeLimit != None:
            if iterationLimit != None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = timeLimit
            self.limitType = 'time'
        else:
            if iterationLimit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'
        self.explorationConstant = explorationConstant        

    # Returns the action needed
    def search(self, initialState):
        self.root = treeNode(initialState, None)

        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
        else:
            for i in range(self.searchLimit):
                self.executeRound()

        bestChild = self.getBestChild(self.root, 0)
        return self.getAction(self.root, bestChild)

    def executeRound(self):
        node = self.selectNode(self.root) # Select a node based on getBestChild
        reward = node.state.getReward() # Get the predicted result of the state
        self.backpropogate(node, reward)

    def selectNode(self, node) -> treeNode:
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                return self.expand(node)
        return node

    def expand(self, node: treeNode) -> treeNode:
        actions = node.state.getPossibleActions()
        for action in actions:
            if action not in node.children.keys():
                newNode = treeNode(node.state.takeAction(action), node)
                node.children[action] = newNode
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True

                newNode.state.evaluate([node.state]) # expand the current node and save state and probabiliies and reward
                return newNode

        raise Exception("Should never reach here")

    def backpropogate(self, node, reward):
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    # Change this to paper impl
    def getBestChild(self, node, explorationValue) -> treeNode:
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():            
            qsa = child.totalReward / child.numVisits

            # The U(s, a) calculation is wrong for now
            # it should be square root of the sum of all actions to get to the current state
            usa = explorationValue * child.state.getProbability() * math.sqrt(child.numVisits) / (1 + child.numVisits)

            nodeValue = qsa + usa
            if nodeValue >= bestValue:
                bestValue = nodeValue
                bestNodes.append(child)
        return random.choice(bestNodes)

    def getAction(self, root, bestChild):
        for action, node in root.children.items():
            if node is bestChild:
                return action
