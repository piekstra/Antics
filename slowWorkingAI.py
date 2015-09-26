  # -*- coding: latin-1 -*-
import random
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Ant import Ant
from Move import Move
from GameState import addCoords
from AIPlayerUtils import *


##
#isValidAttack
#Description: Determines whether the attack with the given parameters is valid
#   Attacking ant is assured to exist and belong to the player whose turn it is
#   **This is a simplified version of the method isValidAttack in Game.py**
#
#Parameters:
#   attackingAnt - The Ant that is attacking (Ant)
#   attackCoord - The coordinates of the Ant that is being attacked ((int,int))
#
#Returns: True if valid attack, or False if invalid attack
##  
def isValidAttack(attackingAnt, attackCoord):
    #we know we have an enemy ant
    range = UNIT_STATS[attackingAnt.type][RANGE]
    diffX = abs(attackingAnt.coords[0] - attackCoord[0])
    diffY = abs(attackingAnt.coords[1] - attackCoord[1])
    
    #pythagoras would be proud
    if range ** 2 >= diffX ** 2 + diffY ** 2:
        #return True if within range
        return True
    else:
        return False

        
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
        self.queen = None
        self.enemyQueen = None
        #simple extremely efficient search agent winner
        super(AIPlayer,self).__init__(inputPlayerId, "SeeSaw") 
    
    
    ##
    # processMove
    # Description: The processMove method looks at the current state
    # of the game and returns a copy of the state that results from
    # making the move
    #
    # Parameters:
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
            enemyAntCoords = [ant.coords for ant in enemyInv.ants]
            # contains the coordinates of ants that the 'moving' ant can attack
            validAttacks = []
            # go through the list of enemy ant locations and check if 
            # we can attack that spot and if so add it to a list of
            # valid attacks (one of which will be chosedn at random)
            for coord in enemyAntCoords:
                if isValidAttack(ant, coord):
                    validAttacks.append(coord)
            # if we can attack, pick a random attack and do it
            if validAttacks:
                attackCoord = random.choice(validAttacks)
                enemyAnt = getAntAt(copyOfState, attackCoord)
                attackStrength = UNIT_STATS[ant.type][ATTACK]
                if enemyAnt.health <= attackStrength:
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
    
    
    def evaluateState(self, currentState):
        # state value ranges between 0 (loser) and 1 (winner!)
        
        # get a reference to the player's inventory
        playerInv = currentState.inventories[currentState.whoseTurn]
        # get a reference to the enemy player's inventory
        enemyInv = currentState.inventories[(currentState.whoseTurn+1) % 2]
        
        if self.queen is None:
            self.queen = playerInv.getQueen()
        if self.enemyQueen is None:
            self.enemyQueen = enemyInv.getQueen()
        
        if self.queen is None:
            return 0.0
        elif self.enemyQueen is None:
            return 1.0
        if playerInv.foodCount >= 11:
            return 1.0
        elif enemyInv.foodCount >= 11:
            return 0.0
        
        # start out state as being neutral ( nobody is winning or losing )
        valueOfState = 0.5
        
        # player is winning more if has more food
        foodDiff = playerInv.foodCount - enemyInv.foodCount
        
        valueOfState += foodDiff * .08
        
        if playerInv.foodCount > enemyInv.foodCount:
            valueOfState += 0.05
        elif playerInv.foodCount < enemyInv.foodCount:
            valueOfState -= 0.05

        # determine how many enemy ants could attack player's queen
        enemiesAdjacentToQueen = []
        for coord in listAdjacent(self.queen.coords):
            ant = getAntAt(currentState, coord)
            if ant and ant.player != currentState.whoseTurn:
                enemiesAdjacentToQueen.append(ant)
        # determine how many ants could attack enemy player's queen
        antsAdjacentToEnemyQueen = []
        for coord in listAdjacent(self.enemyQueen.coords):
            ant = getAntAt(currentState, coord)
            if ant and ant.player == currentState.whoseTurn:
                antsAdjacentToEnemyQueen.append(ant)
        
        # enemy ants around player's queen is BAD
        if len(enemiesAdjacentToQueen) > 0:
            valueOfState -= len(enemiesAdjacentToQueen) * 0.05         
        # player ants around enemy player's queen is GOOD
        if len(antsAdjacentToEnemyQueen) > 0:
            valueOfState += len(antsAdjacentToEnemyQueen) * 0.05
        
        playerFoodCarryAmount = 0
        
        allAnts = getAntList(currentState, currentState.whoseTurn,(DRONE,WORKER,SOLDIER,R_SOLDIER))
        attackAnts = []
        for ant in allAnts:
            if ant.type == WORKER:
                workerAnts.append(ant)
            else:
                attackAnts.append(ant)
        if len(attackAnts) > 2:
            valueOfState -= len(attackAnts) * .2
        if len(workerAnts) > 1:
            return 0.0000000001
        for ant in attackAnts:
            # list of all the distances to enemy ants
            distancesToEnemeies = [stepsToReach(currentState, ant.coords, enemyAnt.coords) for enemyAnt in enemyInv.ants]
            # find the minimum steps to an enemy ant, use that to modify the state's value
            # the fewer the total steps the better
            valueOfState -= (min(distancesToEnemeies) - 1) * .001   
        
        # worker proximity to food or hill/tunnel
        # move ant closer to food than hill/tunnel if not carrying
        # else move ant closer to hill/tunnel than food if carrying
        for ant in workerAnts:
            if ant.carrying:
                valueOfState += .015
                #list of all the distances to player constructions, starting with the hill distance
                distancesToConst = [stepsToReach(currentState, ant.coords, playerInv.getAnthill().coords)]
                #loop through all the tunnel coords and find distance to them, adding them to the distances
                distancesToConst.extend([stepsToReach(currentState, ant.coords, tunnel.coords) for tunnel in playerInv.getTunnels()])
                #find the minimum steps to a construction, use that to modify the state's value
                valueOfState -= (min(distancesToConst) - 1) * .001   
            else:                
                #list of all distances to food
                distancesToFood = []
                for constr in currentState.inventories[NEUTRAL].constrs:
                    if constr.type == FOOD:
                        steps = stepsToReach(currentState, ant.coords, constr.coords)
                        distancesToFood.append(steps)
                valueOfState -= (min(distancesToFood) - 1) * .001
            
        # 4  count amt food ( more the merrier ) (enemy, own) (for example, attacking enemy worker that is carrying is more valuable then attacking enemy worker that is not carrying food)
        # 3  count own carrying ( more is good )
        # 2  check all enemy ants health ( less good )
        # 1  count num ants player has ( lots of ants good )

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
    #coordinates (on their opponent’s side of the board) which represent
    #Locations to place the food sources. This is all that is necessary to 
    #complete the setup phases.
    #
    #Parameters:
    #   currentState - The current state of the game at the time the Game is 
    #       requesting a placement from the player.(GameState)
    #
    #Return: If setup phase 1: list of ten 2-tuples of ints -> [(x1,y1), (x2,y2),…,(x10,y10)]
    #       If setup phase 2: list of two 2-tuples of ints -> [(x1,y1), (x2,y2)]
    ##
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
    ##
    def getMove(self, currentState):
        possibleMoves = listAllLegalMoves(currentState)
        bestValue = 0.0
        bestMove = possibleMoves[0]
        for move in possibleMoves:
            moveValue = self.evaluateState(self.processMove(currentState, move))
            if moveValue > bestValue:
                bestValue = moveValue
                bestMove = move
        return bestMove
    
    
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
