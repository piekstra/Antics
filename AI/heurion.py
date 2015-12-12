# -*- coding: latin-1 -*-

##
# 
# Finale Tournament Submission
#
# Author: Caleb Piekstra
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


    ##
    # __init__
    # Description: Creates a new Player
    #
    # Parameters:
    #   inputPlayerId - The id to give the new player (int)
    ##
    def __init__(self, inputPlayerId):
        
        # whether or not the first move has been made
        self.firstMove = True

        # Heurion
        super(AIPlayer,self).__init__(inputPlayerId, "Heurion") 
            
           
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
    # getBestMove
    # Description: Given a current state, looks at all moves
    #   immediately available and picks the one that has the 
    #   highest-valued resulting state. The neural network is
    #   used to determine the value of a potential state.
    #
    # Parameters:
    #   self - The object pointer
    #   currentState - The state to expand for possible moves
    #
    # Return: The highest scoring move (according to the neural network)
    #
    def getBestMove(self, currentState):
        # return the best move found (highest value resulting state)
        return sorted([(self.evaluateState(self.processMove(currentState, move)), move) for move in listAllLegalMoves(currentState)], reverse=True)[0][1]
            
    
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
        playerInv = currentState.inventories[self.wt]
        # get a reference to the enemy player's inventory
        enemyInv = currentState.inventories[1 - self.wt]
        # get a reference to the enemy's queen
        enemyQueen = enemyInv.getQueen()
        self.dests = [playerInv.getAnthill().coords] + [tunnel.coords for tunnel in playerInv.getTunnels()]
        
        # game over (lost) if player does not have a queen
        #               or if enemy player has 11 or more food
        if playerInv.getQueen() is None or enemyInv.foodCount >= 11 or (len(playerInv.ants) == 1 and playerInv.foodCount == 0):
            return 0.0             
        # game over (win) if enemy player does not have a queen
        #              or if player has 11 or more food
        if enemyQueen is None or playerInv.foodCount >= 11 or (len(enemyInv.ants) == 1 and enemyInv.foodCount == 0):
            return 1.0
        
        # initial state value is neutral ( no player is winning or losing )
        valueOfState = 0.5       
        
        # hurting the enemy queen is a very good state to be in
        valueOfState -= 0.0025 * enemyQueen.health
        valueOfState -= 0.0050 * sum([ant.health for ant in enemyInv.ants])
        valueOfState += 0.0035 * playerInv.foodCount
                
        countWorkers = 0
        # loop through the player's ants and handle rewards or punishments
        # based on whether they are workers or attackers
        for ant in playerInv.ants:
            if ant.type == QUEEN:     
                valueOfState += 0.0026 * ant.health     
                # if the queen is on the hill, this is bad
                if ant.coords in self.dests + self.foods:
                    valueOfState -= 0.2   
                # avoid enemies
                valueOfState += 0.00005 * min(self.vectorDistance(ant.coords, enemyAnt.coords) for enemyAnt in enemyInv.ants)
            elif ant.type == WORKER:    
                countWorkers += 1      
                # if the ant has food,
                # store the distance from the ant to the closest drop-off point  
                if ant.carrying:
                    valueOfState += 0.0030
                    valueOfState -= 0.00005 * min(self.vectorDistance(ant.coords, destCoord) for destCoord in self.dests)
                # if the ant does not have food,
                else:
                    valueOfState -= 0.00005 * min(self.vectorDistance(ant.coords, destCoord) for destCoord in self.foods)
            else:
                # Reward the AI for having ants other than the queen
                valueOfState += 0.005
                # Punish the AI less and less as its ants approach the enemy's queen
                # valueOfState -= 0.0005 * self.vectorDistance(ant.coords, enemyQueen.coords) 
                valueOfState -= 0.00005 * min(self.vectorDistance(ant.coords, enemyAnt.coords) for enemyAnt in enemyInv.ants)      
     
        if countWorkers <= 3:
            # Reward the AI for having workers
            valueOfState += 0.0010 * countWorkers
        else:
            valueOfState -= 0.0050 * countWorkers 
        
        # return the value of the currentState
        return valueOfState   
    

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
            return [(2, 1), (7, 1)] + [(x,3) for x in range(9)]
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
        # on the first move
        if self.firstMove:            
            self.firstMove = False
            self.wt = currentState.whoseTurn
            # keep track of the coordinates of the foods 
            # so they don't have to be re-"found" each time
            self.foods = [item.coords for item in currentState.inventories[NEUTRAL].constrs if item.type == FOOD]        
            
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
        self.firstMove = True
        
        #method template, not implemented
        pass
        
