# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent



class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        currentFoodList = currentGameState.getFood().asList()
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newFoodList = newFood.asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        distFromGhost = manhattan(newPos, newGhostStates[0].getPosition())

        score = 0
        if len(newFoodList) < len(currentFoodList):
            score += 1000
        if len(newFoodList) == 0:
            score += 1000
        else:
            minDist = float('inf')
            for food in newFoodList:
                minDist = min(minDist, manhattan(newPos, food))
            score += -10 * minDist

        if distFromGhost <= 1:
            score += -100000
        else:
            score += 3 * distFromGhost
        if action == Directions.STOP:
            score -= 1000000000

        "*** YOUR CODE HERE ***"
        return score

def manhattan(x, y):
    return abs(x[0]-y[0]) + abs(x[1] - y[1])

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        val = maximize(gameState, self.depth, gameState.getNumAgents() - 1, self.evaluationFunction)
        # print(val)
        return val[1]
        util.raiseNotDefined()

def maximize(state, depth, numGhosts, func):
    if depth == 0 or state.isLose() or state.isWin():
        score = func(state)
        return (score, Directions.STOP)
    else:
        v = float('-inf')
        actions = state.getLegalActions(0)
        bestAction = Directions.STOP
        for action in actions:
            successor = state.generateSuccessor(0, action)
            m = minimize(successor, depth, 1, successor.getNumAgents() - 1, func)
            if m > v:
                v = m
                bestAction = action
        return (v, bestAction)

def minimize(state, depth, ghost, numGhosts, func):
    if state.isLose() or state.isWin():
        score = func(state)
        return score
    v = float('inf')
    actions = state.getLegalActions(ghost)
    for action in actions:
        successor = state.generateSuccessor(ghost, action)
        m = 0
        if ghost < numGhosts:
            m = minimize(successor, depth, ghost + 1, numGhosts, func)
        else:
            m = maximize(successor, depth - 1, numGhosts, func)[0]
        v = min(v, m)
    return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        val = maximizeAB(gameState, self.depth, gameState.getNumAgents() - 1,
                       self.evaluationFunction, float('-inf'), float('inf'))
        return val[1]
        util.raiseNotDefined()

def maximizeAB(state, depth, numGhosts, func, alpha, beta):
    if depth == 0 or state.isLose() or state.isWin():
        score = func(state)
        return (score, Directions.STOP)
    else:
        v = float('-inf')
        actions = state.getLegalActions(0)
        bestAction = Directions.STOP
        for action in actions:
            successor = state.generateSuccessor(0, action)
            m = minimizeAB(successor, depth, 1, successor.getNumAgents() - 1, func, alpha, beta)
            if m > v:
                v = m
                bestAction = action
            alpha = max(alpha, v)
            if alpha > beta:
                break
        return (v, bestAction)

def minimizeAB(state, depth, ghost, numGhosts, func, alpha, beta):
    if state.isLose() or state.isWin():
        score = func(state)
        return score
    v = float('inf')
    actions = state.getLegalActions(ghost)
    for action in actions:
        successor = state.generateSuccessor(ghost, action)
        m = 0
        if ghost < numGhosts:
            m = minimizeAB(successor, depth, ghost + 1, numGhosts, func, alpha, beta)
        else:
            m = maximizeAB(successor, depth - 1, numGhosts, func, alpha, beta)[0]
        v = min(v, m)
        beta = min(beta, v)
        if alpha > beta:
            break
    return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        val = maximizeExp(gameState, self.depth, gameState.getNumAgents() - 1, self.evaluationFunction)
        # print(val)
        return val[1]
        util.raiseNotDefined()

def maximizeExp(state, depth, numGhosts, func):
    if depth == 0 or state.isLose() or state.isWin():
        score = func(state)
        return (score, Directions.STOP)
    else:
        v = float('-inf')
        actions = state.getLegalActions(0)
        bestAction = Directions.STOP
        for action in actions:
            successor = state.generateSuccessor(0, action)
            m = minimizeExp(successor, depth, 1, successor.getNumAgents() - 1, func)
            if m > v:
                v = m
                bestAction = action
        return (v, bestAction)

def minimizeExp(state, depth, ghost, numGhosts, func):
    if depth == 0 or state.isLose() or state.isWin():
        score = func(state)
        return score
    v = 0
    actions = state.getLegalActions(ghost)
    for action in actions:
        successor = state.generateSuccessor(ghost, action)
        m = 0
        if ghost < numGhosts:
            m = minimizeExp(successor, depth, ghost + 1, numGhosts, func)
        else:
            m = maximizeExp(successor, depth - 1, numGhosts, func)[0]
        v += m / len(actions)
    return v

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    food = currentGameState.getFood()
    foodList = food.asList()

    # leftFood = []
    # rightFood = []
    # width = food.width
    # for food in foodList:
    #     if food[0] < (width/3):
    #         leftFood.append(food)
    #     elif food[0] > (2*width/3):
    #         rightFood.append(food)

    numFood = len(foodList)
    pos = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    score = 0

    # Run away from aggressive ghost
    if scaredTimes[0] <= 2:
        distFromGhost = manhattan(pos, ghostStates[0].getPosition())
        if distFromGhost <= 1:
            score += -100000
        # else:
        #     score += 3 * distFromGhost

    # Move towards food
    if numFood == 0:
        score += 1000000
    else:
        distFromFood = float('inf')
        for food in foodList:
            distFromFood = min(distFromFood, manhattan(pos, food))
        score += -10 * distFromFood

    # Collect food
    score += -50 * len(foodList)

    # Move towards capsule
    if scaredTimes[0] > 0:
       score += 10000
    # else:
    #     distFromCapsule = float('inf')
    #     capsules = currentGameState.getCapsules()
    #     for capsule in capsules:
    #         distFromCapsule = min(distFromCapsule, manhattan(pos, capsule))
    #     if distFromCapsule > 1:
    #         score += -1 * distFromCapsule

    # if len(leftFood) == 0:
    #     score += 1000000
    # if len(rightFood) == 0:
    #     score += 1000000

    # # Eat scared ghost

    if scaredTimes[0] > 0:
        distFromGhost = manhattan(pos, ghostStates[0].getPosition())
        if distFromGhost == 1:
            score += 1000
        elif distFromGhost == 0:
            score += 100000
        else:
            score += -12 * distFromGhost

    # Add randomness to prevent getting stuck
    score += random.random() * 10 - 5


    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
