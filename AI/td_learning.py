# -*- coding: latin-1 -*-

##
# 
# Homework 8 - Temporal Difference Learning
#
# Author(s): Caleb Piekstra, Chandler Underwood
#
import random, time, string, argparse, math, sys, types, pickle, collections
import os.path as filePath
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Construction import Construction
from Ant import UNIT_STATS
from Ant import Ant
from Move import Move
from GameState import addCoords
from AIPlayerUtils import *

# Whether to keep learning states and updating utilities
LEARNING = True

##
#AIPlayer
#Description: The responsbility of this class is to interact with the game by
#deciding a valid move based on a given game state. This class has methods that
#will be implemented by students in Dr. Nuxoll's AI course.
#
#Variables:
#   playerId - The id of the player.
##
class AIPlayer(Player):


    ##
    # __init__
    # Description: Creates a new Player
    #
    # Parameters:
    #   inputPlayerId - The id to give the new player (int)
    ##
    def __init__(self, inputPlayerId):        
        
        # initialize the dictionary that will map state tuples to
        # their utilities
        self.utilityDict = {}
        
        #  discount factor for the TD-Learning algorithm
        self.td_lambda = 0.9
        
        # the learning rate for the TD-Learning algorithm
        self.td_alpha = 0.99
        
        # the change rate of the learning rate (higher is more change)
        self.td_alpha_change_rate = 3
        
        # the change amount of the learning rate (higher is more change)
        self.td_alpha_change_amount = 0.1

        # set the unique identifier for the AI and its utilities file (based on the alpha change values)
        self.AIID = str(self.td_alpha_change_rate) + '_' + str(self.td_alpha_change_amount).replace('.', '')+ "_SSFQF"
        
        #file name constant
        self.utilityFile = "utility." + self.AIID 
        
        # keep track of the number of states encountered
        self.statesEncountered = 0
        
        # keep track of how many new states are discovered
        self.newStatesFound = 0
        
        # keep track of how many games have been played
        self.gameCount = 0
        
        # whether or not the first move has been made
        self.firstMove = True
        
        # TODO comments
        if filePath.isfile(self.utilityFile):
            self.loadUtilityList()

        # RTD2 - Relativistic Temporal Differentiator 2.0 
        super(AIPlayer,self).__init__(inputPlayerId, "RTD2-" + self.AIID) 
            
           
    ##
    # vectorDistance
    # Description: Given two cartesian coordinates, determines the 
    #   manhattan distance between them (assuming all moves cost 1)
    #
    # Parameters:
    #   self - The object pointer
    #   pos1 - The first position
    #   pos2 - The second position
    #
    # Return: The manhattan distance
    #
    def vectorDistance(self, pos1, pos2):
        return (abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]))

        
    ##
    # consolidateState
    #
    # Description: Takes a game state and consolidates it
    # to the minimum information needed for learning purposes
    #
    # Notes:get
    #   Dictionary keys are set to single letters to save
    #   space when writing the utilities dict to a file
    #
    # Parameters: 
    #   currentState - the state to compress
    #
    #Return:
    #   the compressed state
    def consolidateState(self, state):

        # get some references for use later
        playerInv = state.inventories[state.whoseTurn]
        enemyInv = state.inventories[1 - state.whoseTurn]
        enemyQueen = enemyInv.getQueen()
        playerQueen = playerInv.getQueen()
        
        #initialize the compressed state dictionary with default values
        compressedState = {}
        
        
        if enemyQueen:
            # keep track of health of enemy queen
            compressedState["e"] = enemyQueen.health
        if playerQueen:
            # keep track of the player queen's health
            compressedState["p"] = playerQueen.health   
        
        # # keep track of the amount of food the player has
        compressedState["f"] = playerInv.foodCount
        
        # keep track of whether the state is a win or a loss state
        if enemyQueen is None or len(enemyInv.ants) == 1 or playerInv.foodCount >= 11:
            compressedState["w"] = True
        elif playerQueen is None or len(playerInv.ants) == 1 or enemyInv.foodCount >= 11:
            compressedState["l"] = True
        else:
            
            antCount = 0
            # check each of the player's ants
            for ant in playerInv.ants:
                if ant.type == WORKER:     
                    antCount += 1
                    
                    # only keep track of a certain number of ants 
                    # any extras don't change the state
                    if antCount > 3:
                        continue
                                       
                    #store the distance from the ant to food
                    if ant.carrying:
                        compressedState[str(antCount)+"c"] = min(self.vectorDistance(ant.coords, destCoord) for destCoord in ([tunnel.coords for tunnel in playerInv.getTunnels()]+[playerInv.getAnthill().coords]))      
                    else:
                        compressedState[str(antCount)] = min(self.vectorDistance(ant.coords, foodCoords) for foodCoords in self.foods) 
            
        # return the evaluation score of the state
        return compressedState
    
         
    ##
    # rewardFunction
    #
    # Description: Determines a reward value based on a compressed state
    #   and returns the reward
    #
    # Parameters:
    #   compressedState - the compressed state dictionary
    #
    # Returns:
    #   The reward value of the state
    ##
    def rewardFunction(self, compressedState):
        if "w" in compressedState:
            return 1.0
        elif "l" in compressedState:
            return -1.0
        return -0.01
        
    
    ##
    # flattenDict
    #
    # Description: Converts a nested dictionary to a 1D dictionary
    # Source: http://stackoverflow.com/questions/6027558/flatten-nested-python-dictionaries-compressing-keys
    #
    # Parameters:
    #   self - the object pointer
    #   d - the dict to flatten
    #   parent_key - the key from one level up in the nested dicts
    #   sep - a seperator for keys and parent keys (default '_')
    #
    # Returns:
    #   a 1D dictionary representing the original nested dict
    ##
    def flattenDict(self, d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(self.flattenDict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    

    ##
    # flattenDictToTuple
    #
    # Description: Converts a nested dictionary to a tuple
    #
    # Parameters:
    #   self - the object pointer
    #   dict - the dict to convert
    #
    # Returns:
    #   a tuple representation of the dict
    ##
    def flattenDictToTuple(self, dict):
        return tuple(sorted(self.flattenDict(dict).items()))
    
        
    ##
    # saveUtilityList
    #
    # Description: Writes the utility list object out to file using pickle library
    #
    ##
    def saveUtilityList(self):
        with open("AI/"+self.utilityFile, 'wb') as f:
            pickle.dump(self.utilityDict, f, 0)
        
        
    ##
    # loadUtilityList
    #
    # Description: Loads the utility list object from file using pickle library
    #
    ##
    def loadUtilityList(self):
        with open(self.utilityFile, 'rb') as f:
            self.utilityDict = pickle.load(f)
    
    
    ## TODO COMMENTS
    def addStateUtility(self, currentState, nextState=None):             
        # convert the current state to tuple form to be used as a dictionary key
        curStateDict = self.consolidateState(currentState)
        curStateTuple = self.flattenDictToTuple(curStateDict) 
        
        # in the case of the first move of a game, if we have not seen
        # the initial state, set it's utility to 0, otherwise leave it
        if nextState is None:
            if curStateTuple not in self.utilityDict:
                self.newStatesFound += 1
                # state not previously encountered, default utility to 0
                self.utilityDict[curStateTuple] = 0
        else:    
            # convert the next state to tuple form to be used as a dictionary key
            nextStateTuple = self.flattenDictToTuple(self.consolidateState(nextState))     
            # store the state's utility in the dict
            if nextStateTuple not in self.utilityDict:
                self.newStatesFound += 1
                # state not previously encountered, default utility to 0
                self.utilityDict[nextStateTuple] = 0
            else:
                self.utilityDict[curStateTuple] += self.td_alpha*(self.rewardFunction(curStateDict) + self.td_lambda*self.utilityDict[nextStateTuple] - self.utilityDict[curStateTuple])
                
        return self.utilityDict[curStateTuple]
    
    
    ## 
    # getBestMove
    # Description: Given a current state, looks at all moves
    #   immediately available and picks the one that has the 
    #   highest utility.
    #
    # Parameters:
    #   self - The object pointer
    #   currentState - The state to expand for possible moves
    #
    # Return: The move with the highest utility
    #
    def getBestMove(self, currentState):
        # holds the best move
        bestMove = None
        # holds the value of the best move
        bestMoveUtil = -99999.0
        
        # holds all legal moves from the currentState
        moves = listAllLegalMoves(currentState)
        
        # loop through all legal moves for the currentState
        for move in moves:
                    
            # get the state that would result if the move is made
            potentialState = self.processMove(currentState, move)
            
            # get the utility of the current state based on the potential state
            stateUtil = self.addStateUtility(currentState, potentialState) 
            
            # keep track of the best move so far and its utility
            if stateUtil > bestMoveUtil:
                bestMoveUtil = stateUtil
                bestMove = move
        
        # introduce a 10% chance of making a random move
        # in order to explore now and then and not always
        # exploit!
        if random.random() < 0.1:
            return moves[random.randint(0, len(moves)-1)]
            
        # return the best move found (highest value resulting state)
        return bestMove
           
    
    ##
    # processMove
    # Description: The processMove method looks at the current state
    # of the game and returns a copy of the state that results from
    # making the move
    #
    # Parameters:
    #   self - The object pointer
    #   currentState - The current state of the game
    #   move - The move which alters the state
    #
    # Return: The resulting state after move is made
    #
    def processMove(self, currentState, move):
        # create a copy of the state (this will be returned
        # after being modified to reflect the move)
        copyOfState = currentState.fastclone()
        
        # get a reference to the player's inventory
        playerInv = copyOfState.inventories[copyOfState.whoseTurn]
        # get a reference to the enemy player's inventory
        enemyInv = copyOfState.inventories[(copyOfState.whoseTurn+1) % 2]
        
        # player is building a constr or ant
        if move.moveType == BUILD:
            # building a constr
            if move.buildType < 0:  
                playerInv.foodCount -= CONSTR_STATS[move.buildType][BUILD_COST]
                playerInv.constrs.append(Construction(move.coordList[0], move.buildType))
            # building an ant
            else: 
                playerInv.foodCount -= UNIT_STATS[move.buildType][COST]
                playerInv.ants.append(Ant(move.coordList[0], move.buildType, copyOfState.whoseTurn))                
        # player is moving an ant
        elif move.moveType == MOVE_ANT:
            # get a reference to the ant
            ant = getAntAt(copyOfState, move.coordList[0])
            # update the ant's location after the move
            ant.coords = move.coordList[-1]
            
            # get a reference to a potential constr at the destination coords
            constr = getConstrAt(copyOfState, move.coordList[-1])
            # check to see if the ant is on a food or tunnel or hill and act accordingly
            if constr:
                # we only care about workers
                if ant.type == WORKER:
                    # if dest is food and can carry, pick up food
                    if constr.type == FOOD:
                        if not ant.carrying:
                            ant.carrying = True
                    # if dest is tunnel or hill and ant is carrying food, ditch it
                    elif constr.type == TUNNEL or constr.type == ANTHILL:
                        if ant.carrying:
                            ant.carrying = False
                            playerInv.foodCount += 1
            # get a list of the coordinates of the enemy's ants                 
            enemyAntCoords = [enemyAnt.coords for enemyAnt in enemyInv.ants]
            # contains the coordinates of ants that the 'moving' ant can attack
            validAttacks = []
            # go through the list of enemy ant locations and check if 
            # we can attack that spot and if so add it to a list of
            # valid attacks (one of which will be chosen at random)
            for coord in enemyAntCoords:
                #pythagoras would be proud
                if UNIT_STATS[ant.type][RANGE] ** 2 >= abs(ant.coords[0] - coord[0]) ** 2 + abs(ant.coords[1] - coord[1]) ** 2:
                    validAttacks.append(coord)
            # if we can attack, pick a random attack and do it
            if validAttacks:
                enemyAnt = getAntAt(copyOfState, random.choice(validAttacks))
                attackStrength = UNIT_STATS[ant.type][ATTACK]
                if enemyAnt.health <= attackStrength:
                    # just to be safe, set the health to 0
                    enemyAnt.health = 0
                    # remove the enemy ant from their inventory (He's dead Jim!)
                    enemyInv.ants.remove(enemyAnt)
                else:
                    # lower the enemy ant's health because they were attacked
                    enemyAnt.health -= attackStrength
        # move ends the player's turn
        elif move.moveType == END:
            # toggle between PLAYER_ONE (0) and PLAYER_TWO (1)
            copyOfState.whoseTurn += 1
            copyOfState.whoseTurn %= 2
        
        # return a copy of the original state, but reflects the move
        return copyOfState
        
    
    ##
    #getPlacement
    #Description: The getPlacement method corresponds to the 
    #action taken on setup phase 1 and setup phase 2 of the game. 
    #In setup phase 1, the AI player will be passed a copy of the 
    #state as currentState which contains the board, accessed via 
    #currentState.board. The player will then return a list of 10 tuple 
    #coordinates (from their side of the board) that represent Locations 
    #to place the anthill and 9 grass pieces. In setup phase 2, the player 
    #will again be passed the state and needs to return a list of 2 tuple
    #coordinates (on their opponent?s side of the board) which represent
    #Locations to place the food sources. This is all that is necessary to 
    #complete the setup phases.
    #
    #Parameters:
    #   currentState - The current state of the game at the time the Game is 
    #       requesting a placement from the player.(GameState)
    #
    #Return: If setup phase 1: list of ten 2-tuples of ints -> [(x1,y1), (x2,y2),?,(x10,y10)]
    #       If setup phase 2: list of two 2-tuples of ints -> [(x1,y1), (x2,y2)]
    #
    def getPlacement(self, currentState):
        numToPlace = 0
        #implemented by students to return their next move
        if currentState.phase == SETUP_PHASE_1:    #stuff on my side
            moves = [(2, 1), (7, 1)] + [(x,3) for x in range(9)]
            self.hillCoords = moves[0]
            return moves
        elif currentState.phase == SETUP_PHASE_2:   #stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]
            
    
    ##
    #getMove
    #Description: The getMove method corresponds to the play phase of the game 
    #and requests from the player a Move object. All types are symbolic 
    #constants which can be referred to in Constants.py. The move object has a 
    #field for type (moveType) as well as field for relevant coordinate 
    #information (coordList). If for instance the player wishes to move an ant, 
    #they simply return a Move object where the type field is the MOVE_ANT constant 
    #and the coordList contains a listing of valid locations starting with an Ant 
    #and containing only unoccupied spaces thereafter. A build is similar to a move 
    #except the type is set as BUILD, a buildType is given, and a single coordinate 
    #is in the list representing the build location. For an end turn, no coordinates 
    #are necessary, just set the type as END and return.
    #
    #Parameters:
    #   currentState - The current state of the game at the time the Game is 
    #       requesting a move from the player.(GameState)   
    #
    #Return: Move(moveType [int], coordList [list of 2-tuples of ints], buildType [int]
    #
    def getMove(self, currentState):
        if self.firstMove:            
            self.firstMove = False
            self.foods = [item.coords for item in currentState.inventories[NEUTRAL].constrs if item.type == FOOD]
        # on each move, add the starting state to the utility dict
        # if it isn't already there
        self.addStateUtility(currentState)            
        return self.getBestMove(currentState)
    
    
    ##
    #getAttack
    #Description: The getAttack method is called on the player whenever an ant completes 
    #a move and has a valid attack. It is assumed that an attack will always be made 
    #because there is no strategic advantage from withholding an attack. The AIPlayer 
    #is passed a copy of the state which again contains the board and also a clone of 
    #the attacking ant. The player is also passed a list of coordinate tuples which 
    #represent valid locations for attack. Hint: a random AI can simply return one of 
    #these coordinates for a valid attack. 
    #
    #Parameters:
    #   currentState - The current state of the game at the time the Game is requesting 
    #       a move from the player. (GameState)
    #   attackingAnt - A clone of the ant currently making the attack. (Ant)
    #   enemyLocation - A list of coordinate locations for valid attacks (i.e. 
    #       enemies within range) ([list of 2-tuples of ints])
    #
    #Return: A coordinate that matches one of the entries of enemyLocations. ((int,int))
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        #Attack a random enemy.
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]
        
     
    ##
    #registerWin
    #Description: The last method, registerWin, is called when the game ends and simply 
    #indicates to the AI whether it has won or lost the game. This is to help with 
    #learning algorithms to develop more successful strategies.
    #
    #Parameters:
    #   hasWon - True if the player has won the game, False if the player lost. (Boolean)
    #
    def registerWin(self, hasWon):
        # set the first move back to True for the next game
        
        if LEARNING:
            # TODO COMMENTS
            prev_alpha = self.td_alpha
            decreaseBy = (1.0 - (1.0/(math.e**(prev_alpha**self.td_alpha_change_rate))))*self.td_alpha_change_amount
            self.td_alpha -= decreaseBy        
            print "AI: " + self.AIID + "  Learning Rate: %0.5f - %0.5f = %0.5f\t\t%d known states\t%d new states" % (prev_alpha, decreaseBy, self.td_alpha, len(self.utilityDict), self.newStatesFound)
            # reset the number of states encountered
            self.statesEncountered = 0        
            # reset how many new states were discovered
            self.newStatesFound = 0
            # increment the number of games played
            self.gameCount += 1
            # save the utilities to a file every 100 games
            if self.gameCount % 100 == 0:
                self.saveUtilityList()
            #method template, not implemented
            self.firstMove = True
        pass
        
