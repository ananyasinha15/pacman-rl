# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util
from pacman_utils.util import flipCoin, manhattanDistance


class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """

        # Pacman's current position
        self.pacmanPosition = state.getPacmanPosition()

        # Tuple of ghost positions
        self.ghostPositions = tuple(state.getGhostPositions())

        #We first calculate the distance to every ghost
        #If this is within 2 steps, its dangerously close, so we highlight this to pacman
        ghostDistances = []
        for g in state.getGhostPositions():
            ghostDistances.append(manhattanDistance(self.pacmanPosition, g))
        self.ghostClose = min(ghostDistances) <= 2  
       
        #We want to calculate the nearest food distance, and how many there are 
        #When there are few pellets left, we want pacman to prioritise reach this
        #because otherwise, the pacman will just continuously avoid the ghost instead 
        foodList = state.getFood().asList()
        if foodList:
            foodDistances = []  
            for f in foodList:
                foodDistances.append(manhattanDistance(self.pacmanPosition, f))
            self.nearestFoodDist = min(foodDistances)
        else:
            self.nearestFoodDist = 0
        self.foodCount = len(foodList)


    def __eq__(self, other):
        """
        Two feature states are equal if their board features match.
        """
        return (
            isinstance(other, GameStateFeatures) and
            self.pacmanPosition == other.pacmanPosition and
            self.ghostPositions == other.ghostPositions and
            self.ghostClose == other.ghostClose and
            self.nearestFoodDist == other.nearestFoodDist and
            self.foodCount == other.foodCount
        )


    
    def __hash__(self):
        """
        The hash of a feature state is the hash of its features. For use as a dictionary key for Q-values and counts. 
        """
        return hash((self.pacmanPosition, self.ghostPositions, self.ghostClose, self.nearestFoodDist, self.foodCount))

class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0

        self.qValues = {}
        self.counts = {}     #keeps track of how many times we've tried a specific move 

        self.prevState = None
        self.prevAction = None

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        "*** YOUR CODE HERE ***"
        return endState.getScore() - startState.getScore()
        

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        "*** YOUR CODE HERE ***"
        return self.qValues.get((state, action), 0.0)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        # Returns the maximum Q-value over all possible actions in a state.
        # Unseen actions default to Q = 0 via getQValue.
        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        return max(self.getQValue(state, a) for a in actions)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update using the following rule:
            Q(state, action) ← Q(state, action) + α(reward +  γ * max_a' Q(nextState, a') - Q(state, action))

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        # Current estimate of Q(state, action)
        currentQValue = self.getQValue(state, action)

        # Estimate of target value
        target = reward + self.getGamma() * self.maxQValue(nextState)

        # Apply the update rule
        newQValue = currentQValue + self.getAlpha() * (target - currentQValue)

        # Store the new Q-value
        self.qValues[(state, action)] = newQValue



    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        "*** YOUR CODE HERE ***"
        self.counts[(state, action)] = self.getCount(state, action) + 1

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        "*** YOUR CODE HERE ***"
        return self.counts.get((state, action), 0)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        # Encourages exploration by prioritising actions tried fewer than maxAttempts times.
        # Otherwise returns the estimated utility(greedy behaviour).

        if counts < self.maxAttempts:
            return float("inf")
        return utility

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """
        training = self.getEpisodesSoFar() < self.getNumTraining()
        stateFeatures = GameStateFeatures(state)

        if training and self.prevState is not None and self.prevAction is not None:
            reward = self.computeReward(self.prevState, state)
            prevStateFeatures = GameStateFeatures(self.prevState)
            self.learn(prevStateFeatures, self.prevAction, reward, stateFeatures)

        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        
        if len(legal) == 0:
            return Directions.STOP
        

        #use epsilon greedy, i.e. with probability epsilon, pick a random action 
        #Prevnts pacman from continuing to exploit when its not useful 
        if training and flipCoin(self.epsilon):
            chosenAction = random.choice(legal)
        else:
            bestScore = float("-inf")
            bestActions = []

            for action in legal:
                qValue = self.getQValue(stateFeatures, action)

                if training: 
                    count = self.getCount(stateFeatures, action)
                    score = self.explorationFn(qValue, count)
                else:
                    score = qValue
                
                if score > bestScore:
                    bestScore = score
                    bestActions = [action]
                elif score == bestScore:
                    bestActions.append(action)
            
            #this needs to be in this clause because otherwise bestActions is never defined,
            #but still gets called
            chosenAction = random.choice(bestActions)

        if training:
            self.updateCount(stateFeatures, chosenAction)
        
        self.prevState = state
        self.prevAction = chosenAction

        return chosenAction

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        # Learn from the final transition into a win/loss state
        if self.prevState is not None and self.prevAction is not None:
            reward = self.computeReward(self.prevState, state)
            self.learn(GameStateFeatures(self.prevState), self.prevAction, reward, GameStateFeatures(state))

        # Reset per-episode transition memory
        self.prevState = None
        self.prevAction = None

        print(f"Game {self.getEpisodesSoFar()} just ended!")

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
