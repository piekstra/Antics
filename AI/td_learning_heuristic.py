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

#file name constant
UTILITY_FILE = "underwoo16_piekstra17_utility.awesome"

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
        
        # TODO comments
        if filePath.isfile(UTILITY_FILE):
            self.loadUtilityList()

        # RTD2 - Relativistic Temporal Differentiator 2.0 
        super(AIPlayer,self).__init__(inputPlayerId, "RTD2") 
            
           
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
        
        #keep track of the number of ants for dictionary reference
        antCount = 0

        # check each of the player's ants
        for ant in playerInv.ants:
            if ant.type == QUEEN:
                compressedState["queen"] = {}
                # keep track of if the queen exists and is on hill
                compressedState["queen"]["exists"] = True
                # keep track of the queen's health
                compressedState["queen"]["health"] = playerQueen.health
                
                #whether or not the queen is on the hill
                if ant.coords == self.hillCoords:
                    compressedState["queen"]["on_hill"] = True
                else:
                    compressedState["queen"]["on_hill"] = False
            
            #for each non-queen ant
            else:
            
                #set the existence variable to true
                antCount += 1
                
                workerID = "ant" + str(antCount)
                compressedState[workerID] = {}
                compressedState[workerID]["exists"] = True
                
                #store the distance from the worker to enemy queen
                if enemyQueen:
                    compressedState[workerID]["distance"] = (abs(ant.coords[0] - enemyQueen.coords[0]) +
                                                          abs(ant.coords[1] - enemyQueen.coords[1]))
                else:
                    #if enemy queen does not exist, default to zero
                    compressedState[workerID]["distance"] = 0

        if enemyQueen:
            compressedState["enemy_queen"] = {}
            #keep track of existence of enemy queen
            compressedState["enemy_queen"]["exists"] = True
            # keep track of health of enemy queen
            compressedState["enemy_queen"]["health"] = enemyQueen.health
        else:
            compressedState["won"] = True
            # #keep track of existence of enemy queen
            # compressedState["enemy_queen"]["exists"] = False
            # # keep track of health of enemy queen
            # compressedState["enemy_queen"]["health"] = 0
        if playerQueen is None or antCount == 0:
            compressedState["lost"] = True
            
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
    #   A reward value between -1.0 and 1.0 inclusive
    ##
    def rewardFunction(self, compressedState):
        if "won" in compressedState:
            return 1.0
        elif "lost" in compressedState:
            return -1.0
        reward = 0    
        enemyQueenHealthWeight = 0.1
        queenHealthWeight = 0.01
        nonQueenAntDistWeight = 0.001
        for key in compressedState.keys():
            if key == "enemy_queen":
                reward -= enemyQueenHealthWeight*compressedState[key]["health"]
            if key == "queen":
                reward -= queenHealthWeight*(UNIT_STATS[QUEEN][HEALTH] - compressedState[key]["health"])
            elif key.startswith("ant"):
                reward += nonQueenAntDistWeight*(20 - compressedState[key]["distance"])
        return reward
            
    
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
        with open("AI/"+UTILITY_FILE, 'wb') as f:
            pickle.dump(self.utilityDict, f, 0)
        
        
    ##
    # loadUtilityList
    #
    # Description: Loads the utility list object from file using pickle library
    #
    ##
    def loadUtilityList(self):
        with open(UTILITY_FILE, 'rb') as f:
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
                # state not previously encountered, default utility to 0
                self.utilityDict[curStateTuple] = 0
        else:    
            # convert the next state to tuple form to be used as a dictionary key
            nextStateTuple = self.flattenDictToTuple(self.consolidateState(nextState))     
            # store the state's utility in the dict
            if nextStateTuple not in self.utilityDict:
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
        # loop through all legal moves for the currentState
        for move in listAllLegalMoves(currentState):
                    
            # get the state that would result if the move is made
            potentialState = self.processMove(currentState, move)
            
            # get the utility of the current state based on the potential state
            stateUtil = self.addStateUtility(currentState, potentialState) 
            
            # keep track of the best move so far and its utility
            if stateUtil > bestMoveUtil:
                bestMoveUtil = stateUtil
                bestMove = move
                
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
    # mapStateToArray
    # Description: Given a state, looks at specific information
    #   contained in the state and generates floating point 
    #   representations of these between 0.0 and 1.0. These
    #   values are the appended to an array and returned.
    #
    # Parameters:
    #   self - The object pointer
    #   state - The current state of the game
    #
    # Return: An array of floats containing information about
    #   the state.
    #
    def mapStateToArray(self, state):
        # get a reference to the player's inventory
        playerInv = state.inventories[state.whoseTurn]
        # get a reference to the enemy player's inventory
        enemyInv = state.inventories[(state.whoseTurn+1) % 2]
        # get a reference to the enemy's queen
        enemyQueen = enemyInv.getQueen()
        # get a reference to the player's queen
        playerQueen = playerInv.getQueen()
        
        # create the array to represent certain data in the state
        stateArray = []
        
        # queen on hill -> 0 or 1
        stateArray.append(float(playerQueen.coords == self.hillCoords))
        
        # how many ants (max 2): number of non-queen ants / 2.0
        #   -> (0 0.5 1)
        stateArray.append((len(playerInv.ants) - 1) / 2.0)
        
        # The following if/else covers determining the distances
        # from each non-queen ant to the enemy queen
        # max distance is roughly 20 (by squares)
        #   float between 0 and 1 <- (distance/20)
        # set default distances (in case ants don't exist)
        if enemyQueen is None:
            # NO ENEMY QUEEN:
            # distance of ants from enemy queen is 0
            stateArray.append(0.0)
            stateArray.append(0.0)
            # enemy queen's health is 0
            stateArray.append(0.0)
            # player wins (enemy queen dead) -> 1.0
            stateArray.append(1.0)
        else:
            # if there is an enemy queen, default ant distances are 1
            stateArray.append(1.0)
            stateArray.append(1.0)
            
            # initial idx for the first non-queen ant's distance
            antDistanceIdx = 2
            for ant in playerInv.ants:
                if ant != playerQueen:
                    stateArray[antDistanceIdx] = self.vectorDistance(ant.coords, enemyQueen.coords) / 20.0
                    antDistanceIdx += 1
                    # just in case there are magically more than 2 non queen ants, break early
                    if antDistanceIdx >= 4:
                        break
            
            # enemy queen health 
            #   -> enemyQueen.health / 4
            stateArray.append(enemyQueen.health / float(UNIT_STATS[QUEEN][HEALTH]))
            
            # if player has lost, set the "win/loss" value to 0.0
            if playerQueen is None or enemyInv.foodCount >= 11:
                stateArray.append(0.0)
            # if player has won, set the "win/loss" value to 1.0
            elif enemyQueen is None or playerInv.foodCount >= 11:
                stateArray.append(1.0)
            # haven't won or loss yet, keep "win/loss" value at 0.5
            else:
                stateArray.append(0.5)

        # add bias (constant)
        stateArray.append(self.bias)
        
        # save the array as the neural network's inputs
        self.neuralNetInput = stateArray
        # return the array representation of the state
        return stateArray
    
    
    ##
    # evaluateState
    # Description: The evaluateState method looks at a state and
    # assigns a value to the state based on how well the game is
    # going for the current player
    #
    # Parameters:
    #   self - The object pointer
    #   currentState - The state to evaluate
    #
    # Return: The value of the state on a scale of 0.0 to 1.0
    # where 0.0 is a loss and 1.0 is a victory and 0.5 is neutral
    # (neither winning nor losing)
    #
    # Direct win/losses are either a technical victory or regicide
    #
    def evaluateState(self, currentState):
        # get a reference to the player's inventory
        playerInv = currentState.inventories[currentState.whoseTurn]
        # get a reference to the enemy player's inventory
        enemyInv = currentState.inventories[(currentState.whoseTurn+1) % 2]
        # get a reference to the enemy's queen
        enemyQueen = enemyInv.getQueen()
        
        # game over (lost) if player does not have a queen
        #               or if enemy player has 11 or more food
        if playerInv.getQueen() is None or enemyInv.foodCount >= 11:
            return 0.0             
        # game over (win) if enemy player does not have a queen
        #              or if player has 11 or more food
        if enemyQueen is None or playerInv.foodCount >= 11:
            return 1.0
        
        # initial state value is neutral ( no player is winning or losing )
        valueOfState = 0.5       
        
        # hurting the enemy queen is a very good state to be in
        valueOfState += 0.025 * (UNIT_STATS[QUEEN][HEALTH] - enemyQueen.health)
                
        # loop through the player's ants and handle rewards or punishments
        # based on whether they are workers or attackers
        for ant in playerInv.ants:
            if ant.type == QUEEN:     
                # if the queen is on the hill, this is bad
                if ant.coords == self.hillCoords:
                    return 0.001
            else:
                # Reward the AI for having ants other than the queen
                valueOfState += 0.01
                # Punish the AI less and less as its ants approach the enemy's queen
                valueOfState -= 0.005 * self.vectorDistance(ant.coords, enemyQueen.coords) 
            
        # return the value of the currentState
        return valueOfState     
    
    
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
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            # save the hill coords
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
        
        # TODO COMMENTS
        x = self.td_alpha
        # decreaseBy = (1.0 - (1.0/(math.e**(x**16))))*0.1
        decreaseBy = 0.001 if x > 0.1 else 0
        self.td_alpha -= decreaseBy
        print "LR-B4: %0.5f\tLR-AF: %0.5f\tDelta: %0.5f\tNum States Encountered: %d" % (x, self.td_alpha, decreaseBy, len(self.utilityDict))
        self.saveUtilityList()
        
        #method template, not implemented
        pass
        
