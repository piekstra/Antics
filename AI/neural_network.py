# -*- coding: latin-1 -*-

##
# 
# Homework 7 - Artificial Neural Network
#
# Author(s): Caleb Piekstra, Max Robinson
#
import random, time, string, argparse, math, sys, types
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Construction import Construction
from Ant import UNIT_STATS
from Ant import Ant
from Move import Move
from GameState import addCoords
from AIPlayerUtils import *


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


    #__init__
    #Description: Creates a new Player
    #
    #Parameters:
    #   inputPlayerId - The id to give the new player (int)
    ##
    def __init__(self, inputPlayerId):
        # learning rate of the neural network
        self.alpha = 0.8
        
        # the file to write out matrices to when a streak
        # of acceptable evaluations has occurred
        self.matrixOutputFile = "network_results_a08.txt"
        
        # the constant bias value
        self.bias = 1.0
        
        # keep track of the number of successful 10k consecutive
        # acceptable results
        self.acceptableResultsCount = 0
        
        # consecutive acceptable results streak number
        self.acceptableStreak = 1000
        
        # initialize a list to determine how many consecutive 
        # times the network produced an acceptable result
        # (within 0.03  of actual)       
        # a 0 means unacceptable, a 1 means acceptable
        # if the sum of the list is equal to the acceptableStreak
        # than the neural network is providing accurate evaluations
        # on average
        self.acceptableResults = [0]*self.acceptableStreak
        
        # initialize the network's input array values (0.0 by default)
        self.neuralNetInput = [0.0]*7
    
        # initialize the hidden layer weight matrix (7 x 12)
        # commented out when learning stage is complete
        # self.hiddenLayerWeights = [[1.0]*12 for _ in range(7)]
        
        # initialize the hidden layer's weights based on the learned values
        self.hiddenLayerWeights = [
            [3.200784, 3.200784, 3.200784, 3.200784, 3.200784, 3.200784, 3.200784, 3.200784, 3.200784, 3.200784, 3.200784, 3.200784], 
            [0.309659, 0.309659, 0.309659, 0.309659, 0.309659, 0.309659, 0.309659, 0.309659, 0.309659, 0.309659, 0.309659, 0.309659], 
            [0.129364, 0.129364, 0.129364, 0.129364, 0.129364, 0.129364, 0.129364, 0.129364, 0.129364, 0.129364, 0.129364, 0.129364], 
            [0.155654, 0.155654, 0.155654, 0.155654, 0.155654, 0.155654, 0.155654, 0.155654, 0.155654, 0.155654, 0.155654, 0.155654], 
            [0.244135, 0.244135, 0.244135, 0.244135, 0.244135, 0.244135, 0.244135, 0.244135, 0.244135, 0.244135, 0.244135, 0.244135], 
            [0.010028, 0.010028, 0.010028, 0.010028, 0.010028, 0.010028, 0.010028, 0.010028, 0.010028, 0.010028, 0.010028, 0.010028], 
            [-0.979944, -0.979944, -0.979944, -0.979944, -0.979944, -0.979944, -0.979944, -0.979944, -0.979944, -0.979944, -0.979944, -0.979944]]
            
        # initialize the output layer weight matrix (13 x 1)
        # 13 is (number of hidden layer perceptrons + 1 for bias)
        # commented out when learning stage is complete
        # self.outputLayerWeights = [[1.0] for _ in range(13)]
        
        # initialize the output layer's weights based on the learned values
        self.outputLayerWeights = [[-0.891793], [-0.891793], [-0.891793], [-0.891793], [-0.891793], [-0.891793], [-0.891793], [-0.891793], [-0.891793], [-0.891793], [-0.891793], [-0.891793], [4.251797]]
        
        # initialize the hidden layer's perceptron outputs (1 x 12)
        self.hiddenLayerOutputs =[[1.0]*12]
        
        # initialize the neural network output
        self.networkOutput = 0.0

        # Neo - The 1.0 (One)
        super(AIPlayer,self).__init__(inputPlayerId, "Neo") 
            
           
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
    # matrixMult
    # Description: Multiplies two matrices and returns the 
    #   resulting matrix
    #
    # Parameters:
    #   self - The object pointer
    #   matrix1 - The first matrix
    #   matrix2 - The second matrix
    #
    # Notes:
    #   An n x m matrix has n rows and m columns
    #
    #   Both matrix1 and matrix2 should be 2D lists. The returned
    #   matrix result will also be a 2D list.
    #
    #   Example1: 
    #       multiplying a 1x3 by a 3x3 produces a 1x3
    #       as follows:
    #       [[1,2,3]] * [[1,1,1],[2,2,2],[0,1,2]] 
    #           = [[5,8,11]]
    #   
    #   Example2: 
    #       multiplying a 3x2 by a 2x2 produces a 3x2
    #       as follows:
    #       [[1,2],[2,1],[1,1]] * [[0,1],[1,2]] 
    #           = [[2,5], [1,4], [1,3]]
    #
    # Return: The matrix resulting from multiplying matrix1 with matrix2
    # 
    def matrixMult(self, matrix1, matrix2):
        multMatrix = []
        for row in matrix1:
            multRow = []
            for col in self.transpose(matrix2):
                # multiply each element in matrix1's row with the corresponding element in
                # matrix2's col and sum the results
                multRow.append(sum([rowEl*colEl for rowEl,colEl in zip(row,col)]))
            multMatrix.append(multRow)
        return multMatrix
        
    
    ## returns the transpose of the matrix
    ## 
    # transpose
    # Description: Finds the transpose of a matrix
    #
    # Parameters:
    #   self - The object pointer
    #   matrix - The matrix to transpose
    #   matrix2 - The second matrix
    #
    # Notes:
    #   The transpose of a matrix is a new matrix where
    #   the rows and columns are swapped
    #
    #   Example1: 
    #       transpose([[1,1,1],[2,2,2],[3,3,3]])
    #           = [[1,2,3],[1,2,3],[1,2,3]]
    #
    def transpose(self, matrix):
        return [list(row) for row in zip(*matrix)]
    
    
    ## TODO
    # applies g function to resulting matrix from mult
    def gMatrixMult(self, matrix1, matrix2):
        multResult = self.matrixMult(matrix1, matrix2)
        return [[1/(1 + math.e**-el) for el in multResult[0]]]
    
    
    ## TODO
    def getBestMove(self, currentState):
        # holds the best move
        bestMove = None
        # holds the value of the best move
        bestMoveVal = 0.0
        # loop through all legal moves for the currentState
        for move in listAllLegalMoves(currentState):
            # get the state that would result if the move is made
            potentialState = self.processMove(currentState, move)
            
            # get the value of the resulting state via the neural network
            resultingStateVal = self.theMatrix(self.mapStateToArray(potentialState))  
            
            
            # if the resulting state is a goal state (win) just return the move
            if resultingStateVal == 1.0:
                return move
            # otherwise keep track of the best move so far and it's value
            if resultingStateVal > bestMoveVal:
                bestMoveVal = resultingStateVal
                bestMove = move
        return bestMove
        
        
    ## TODO
    def getBestMoveWithPropogation(self, currentState):
        # holds the best move
        bestMove = None
        # holds the value of the best move
        bestMoveVal = 0.0
        # loop through all legal moves for the currentState
        for move in listAllLegalMoves(currentState):
            # get the state that would result if the move is made
            potentialState = self.processMove(currentState, move)
            
            # All of the commented out sections below are because the network
            # has been trained and is no longer learning
            resultingStateVal = self.theMatrix(self.mapStateToArray(potentialState))       
            
            # get the value of the resulting state via the evaluation function
            resultingStateVal = self.theMatrix(self.mapStateToArray(potentialState))  #self.evaluateState(potentialState)
            # get the value of the resulting state via the neural network
            neuralNetworkResult = self.theMatrix(self.mapStateToArray(potentialState))              
            # network error
            glitchInMatrix = resultingStateVal - neuralNetworkResult
            # determine whether the error was acceptable or not
            acceptableError = abs(glitchInMatrix) < 0.03
            # remove the last acceptable result from the list (far left)
            self.acceptableResults.pop(0)
            # append the latest acceptable result to the list (far right)            
            self.acceptableResults.append(int(acceptableError))
            
            # if the last 10000 consecutive state evaluations were acceptable, 
            # the network's results are stable
            if sum(self.acceptableResults) == self.acceptableStreak:
                self.acceptableResultsCount += 1
                self.acceptableResults = [0]*self.acceptableStreak
                # write weight matrices to file
                with open(self.matrixOutputFile, 'a') as file:
                    file.write("Hidden layer weights %d:\n" % self.acceptableResultsCount)
                    for row in self.hiddenLayerWeights:
                        for weight in row:
                            file.write("%.6f\t" % weight)
                        file.write("\n")
                    file.write("\nOutput layer weights %d:\n" % self.acceptableResultsCount)
                    for row in self.outputLayerWeights:
                        for weight in row:
                            file.write("%.6f\t" % weight)
                        file.write("\n")
                    file.write("\n")  
            
            # perform backpropogation to teach the neural network
            self.backpropogation(resultingStateVal, neuralNetworkResult)
            
            if resultingStateVal == 1.0:
                return move
            if resultingStateVal > bestMoveVal:
                bestMoveVal = resultingStateVal
                bestMove = move
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
    # TODO
    def mapStateToArray(self, state):
        # get a reference to the player's inventory
        playerInv = state.inventories[state.whoseTurn]
        # get a reference to the enemy player's inventory
        enemyInv = state.inventories[(state.whoseTurn+1) % 2]
        # get a reference to the enemy's queen
        enemyQueen = enemyInv.getQueen()
        # get a reference to the player's queen
        playerQueen = playerInv.getQueen()
        
        ## State Array Creation
        # create the array to represent the state (default all 0's)
        stateArray = []
        
        # queen on hill -> 0 or 1
        stateArray.append(float(playerQueen.coords == self.hillCoords))
        
        # how many ants: number of non-queen ants / 2.0
        #   -> (0 0.5 1)
        stateArray.append((len(playerInv.ants) - 1) / 2.0)
        
        # distance from each non-queen ant to enemy queen
        # max distance is roughly 20 (by squares)
        #   float between 0 and 1 (distance/20)
        # set default distances (in case ants don't exist)
        if enemyQueen is None:
            # if no enemy queen, default distances are 0
            stateArray.append(0.0)
            stateArray.append(0.0)
            stateArray.append(0.0)
            stateArray.append(0.5)
        else:
            # if there is an enemy queen, default distances are 1
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
            
            # if player has lost, set the "win/loss" value to 0
            if playerQueen is None or enemyInv.foodCount >= 11:
                stateArray.append(0.0)
            # if player has won, set the "win/loss" value to 1
            elif enemyQueen is None or playerInv.foodCount >= 11:
                stateArray.append(1.0)
            else:
                stateArray.append(0.5)

        # set bias (constant)
        stateArray.append(self.bias)
        
        # print stateArray
        self.neuralNetInput = stateArray
        # return the array representation of the state
        return stateArray
        
        
    ## 
    # TODO
    #
    # !!!!!!!!! ROW x COL !!!!!!!!!!!!!!!!!!
    # 2x4 matrix:
    # row1 [col1, col2, col3, col4]
    # row2 [col1, col2, col3, col4]
    #    
    #    3x4
    #    [1, 2, 3, 4]
    #    [1, 2, 3, 4]
    #    [1, 2, 3, 4]
    #
    # Return: The value of the state on a scale of 0.0 to 1.0
    # where 0.0 is a loss and 1.0 is a victory and 0.5 is neutral
    # (neither winning nor losing)
    #
    def theMatrix(self, stateArray):
        # mult input x hidden layer weights
        # note that state array becomes a 2D array
        # so that matrix mult works (it requires 2D arrays)
        hiddenLayerValues = self.gMatrixMult([stateArray], self.hiddenLayerWeights)
        
        # add a bias value (constant) (now a 1x13)
        hiddenLayerValues[0].append(self.bias)
        
        # save the hidden layer's outputs
        self.hiddenLayerOutputs = hiddenLayerValues
        # print "hidden layer values", hiddenLayerValues
        
        # mult hidden layer values x output layer weights
        outputLayerValues = self.gMatrixMult(hiddenLayerValues, self.outputLayerWeights)
        
        # save the network's output
        self.networkOutput = outputLayerValues[0][0]
        
        # print outputLayerValues[0][0]        
        return outputLayerValues[0][0]
    
    
    ## TODO
    # targetVal from evaluateState
    # actualVal from theMatrix (neural network)
    def backpropogation(self, targetVal, actualVal):
        # output perceptron error
        err = targetVal - actualVal        
        delta = actualVal*(1-actualVal)*err
        
        ## hidden layer perceptron errors
        errs = [weight[0]*delta for weight in self.outputLayerWeights]
        # print self.outputLayerWeights, self.hiddenLayerOutputs, errs
        deltas = [b*(1-b)*errs[idx] for idx, b in enumerate(self.hiddenLayerOutputs[0])]
        
        ## 
        # Adjust each weights in the network:  W_ij = W_ij + alpha * delta_j * x_i where:
        # W_ij is the weight between nodes i and j
        # alpha is a learning rate
        # delta_j is the error term for node j
        # x_i  is the input that the weight was applied to     
        #        
        ## output layer weights
        self.outputLayerWeights = [[weight[0] + self.alpha*delta*self.hiddenLayerOutputs[0][idx]] for idx, weight in enumerate(self.outputLayerWeights)]
        
        ## hidden layer weights
        for idx_i, row in enumerate(self.hiddenLayerWeights):
            for idx_j, weight in enumerate(row):
                self.hiddenLayerWeights[idx_i][idx_j] += self.alpha*deltas[idx_j]*self.neuralNetInput[idx_i]
        
    
    ##
    # evaluateevaluateState
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
        #method templaste, not implemented
        pass
        
