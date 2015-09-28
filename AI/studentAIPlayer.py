  # -*- coding: latin-1 -*-
import random
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Construction import Construction
from Ant import UNIT_STATS
from Ant import Ant
from Move import Move
from GameState import addCoords
from AIPlayerUtils import *


# a representation of a 'node' in the search tree
treeNode = {
    "move"              : None,
    "potential_state"   : None,
    "state_value"       : 0.0,
    "parent"            : None
}

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
        # a depth limit for the search algorithm
        self.maxDepth = 2
        # InfoSeargent - the top scecret military INFOrmed SEARch aGENT
        super(AIPlayer,self).__init__(inputPlayerId, "InfoSeargent") 
    
    
    ##
    # evaluateNodes
    # Description: The evaluateNodes method evaluates a list of nodes
    # and determines their overall evaluation score.
    #
    # Parameters:
    #   self - The object pointer
    #   nodes - The list of nodes to evaluate
    #
    # Return: An overall evaluation score of the list of nodes
    #
    def evaluateNodes(self, nodes):
        bestValue = 0.0
        for node in nodes:
            if node["state_value"] > bestValue:
                bestValue = node["state_value"]
        return bestValue
    
    
    ##
    # exploreTree
    # Description: The exploreTree method explores the search tree
    # and returns the Move that leads to the best branch in the tree
    # along with a score indicating just how 'good' that branch is
    # This method is recursive.
    #
    # Parameters:
    #   self - The object pointer
    #   currentState - The current state being 'searched'
    #   playerId - The agent's player id (whose turn it is for now)
    #   currentDepth - The depth the given state is in the search tree
    #
    # Return: Either the best Move to make in the given state or an
    #   evaluation score for that move on the range [0.0...1.0] (based
    #   on the current depth)
    #
    def exploreTree(self, currentState, playerId, currentDepth):
        nodeList = []
        possibleMoves = listAllLegalMoves(currentState)
        #parentNode = nodeList[len(nodeList)-1]        
        bestScore = 0.0
        bestNode = None
        for move in possibleMoves:
            resultingState = self.processMove(currentState, move)
            resultingState.whoseTurn = playerId
            for inv in resultingState.inventories[PLAYER_ONE:NEUTRAL]:
                for ant in inv.ants:
                    ant.hasMoved = False
            newNode = treeNode.copy()
            newNode["move"] = move
            newNode["potential_state"] = resultingState
            newNode["state_value"] = self.evaluateState(resultingState)
            # if we have found a winning state, do not continue
            # to evaluate the other moves
            if newNode["state_value"] == 1.0:
                print "found goal state!"
                bestScore = 1.0
                bestNode = newNode
                break
            #newNode["parent"] = parentNode
            nodeList.append(newNode)
        # base case
        if currentDepth == self.maxDepth:
            return self.evaluateNodes(nodeList)
        # recursive case
        else:
            # if the bestNode was already found, do not bother
            # recursively re-scoring each node
            if bestNode is None:
                for node in nodeList:
                    print "exploring move, initial value:", node["state_value"]
                    node["state_value"] = self.exploreTree(node["potential_state"], playerId, currentDepth+1)
                    print "final value:", node["state_value"]
                    if node["state_value"] > bestScore:
                        bestScore = node["state_value"]
                        bestNode = node
        return bestNode["move"]
    
    
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
    def evaluateState(self, currentState):
        # state value ranges between 0 (loser) and 1 (winner!)
        
        # get a reference to the player's inventory
        playerInv = currentState.inventories[currentState.whoseTurn]
        # get a reference to the enemy player's inventory
        enemyInv = currentState.inventories[(currentState.whoseTurn+1) % 2]
        # get a reference to the player's queen
        queen = playerInv.getQueen()
        # get a reference to the enemy's queen
        enemyQueen = enemyInv.getQueen()
        
        # game over (lost) if player does not have a queen
        #               or if enemy player has 11 or more food
        if queen is None or enemyInv.foodCount >= 11:
            return 0.0
        # game over (win) if enemy player does not have a queen
        #              or if player has 11 or more food
        if enemyQueen is None or playerInv.foodCount >= 11:
            return 1.0
        
        # start out state as being neutral ( nobody is winning or losing )
        valueOfState = 0.5        
        # player is winning more if has more food
        valueOfState += 0.04 * (playerInv.foodCount - enemyInv.foodCount)
        # hurting the enemy queen is a very good state to be in
        valueOfState += 0.1 * (UNIT_STATS[QUEEN][HEALTH] - enemyQueen.health)
        
        # the number of workers belonging to the player
        workerCount = 0
        # loop through the player's ants and handle rewards or punishments
        # based on whether they are workers or attackers
        for ant in playerInv.ants:
            if ant.type == WORKER:
                workerCount += 1
                # Punish the AI severely for having more than 1 worker
                if workerCount > 1:       
                    return 0.001       
                # the distance to the objective closest to the worker
                minObjectiveDist = 99       
                # handle worker objectives differently based on .carrying status
                if ant.carrying:
                    # Reward the AI for every worker ant that is carrying food
                    valueOfState += .015
                    # the distance to the constr closest to the worker
                    minObjectiveDist = 99
                    # look through all of the player's constrs and find the distance to the closest one
                    for constr in playerInv.constrs:
                        if constr.type == ANTHILL or constr.type == TUNNEL:
                            constrDist = stepsToReach(currentState, ant.coords, constr.coords)
                            if constrDist < minObjectiveDist:
                                minObjectiveDist = constrDist
                    # Punish the AI less and less as the worker approaches the closest constr
                    valueOfState -= .001 * (minObjectiveDist - 1) 
                else:                
                    for constr in currentState.inventories[NEUTRAL].constrs:
                        # the player should not try to go for the enemy's food
                        if constr.type == FOOD and constr.coords[1] <= 3:
                            # keep track of the closest food
                            foodDist = stepsToReach(currentState, ant.coords, constr.coords)
                            if foodDist < minObjectiveDist:
                                minObjectiveDist = foodDist
                    # Punish the AI less and less as the worker approaches the closest food
                    valueOfState -= (minObjectiveDist - 1) * .001
            elif ant.type == QUEEN:
                # Punish the AI severely for leaving the queen on a constr
                if getConstrAt(currentState, ant.coords) is not None:      
                    return 0.001
            else:
                # Reward the AI for having attack ants
                valueOfState += 0.05
                # Punish the AI less and less as it's attack ants approach the enemy's queen
                valueOfState -= 0.02 * stepsToReach(currentState, ant.coords, enemyQueen.coords) 
        
        # ensure that 0.0 is a loss and 1.0 is a win ONLY
        if valueOfState < 0.0:
            return 0.001 + (valueOfState * 0.0001)
        if valueOfState > 1.0:
            return 0.999
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
    #coordinates (on their opponent�s side of the board) which represent
    #Locations to place the food sources. This is all that is necessary to 
    #complete the setup phases.
    #
    #Parameters:
    #   currentState - The current state of the game at the time the Game is 
    #       requesting a placement from the player.(GameState)
    #
    #Return: If setup phase 1: list of ten 2-tuples of ints -> [(x1,y1), (x2,y2),�,(x10,y10)]
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
        return self.exploreTree(currentState, currentState.whoseTurn, 1)
    
    
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

 
## UNIT TEST(S) 
# imports required for the unit test(s)
from GameState import *
from Inventory import *
from Location import *

# create a game board
board = [[Location((col, row)) for row in xrange(0,BOARD_LENGTH)] for col in xrange(0,BOARD_LENGTH)]

# create player 1's inventory
p1Inventory = Inventory(PLAYER_ONE, [], [], 0)

# Give the player a worker ant for move testing
p1Inventory.ants.append(Ant((0,0), WORKER, PLAYER_ONE))
# Make sure to give player a queen so enemy does not
# instantly win
p1Inventory.ants.append(Ant((1,1), QUEEN, PLAYER_ONE))

# create player 2's inventory
p2Inventory = Inventory(PLAYER_TWO, [], [], 0)

# Make sure to give the enemy a queen so player does not
# instantly win
p2Inventory.ants.append(Ant((1,1), QUEEN, PLAYER_TWO))

# create a neutral inventory (food!)
neutralInventory = Inventory(NEUTRAL, [], [], 0)

# create a basic game state
gameState = GameState(board, [p1Inventory, p2Inventory, neutralInventory], MENU_PHASE, PLAYER_ONE)

# create a move to move the player's WORKER to Location (0, 1)
move = Move(MOVE_ANT, [(0,0), (0,1)], None)

# Create the SeeSaw AI 
aiPlayer = AIPlayer(PLAYER_ONE)

# Provess the move and save the copy of the state after the move is made
newState = aiPlayer.processMove(gameState, move)

# verify that the ANT was moved to Location (0, 1)
if getAntAt(newState, (0,1)):
    # get the 'value' of the state resulting from making the move
    stateValue = aiPlayer.evaluateState(newState)
    
    # check the value of the new state
    # note that minObjectiveDist defaults to 99
    # the value should be:
    #    NEUTRAL (0.5) minus (minObjectiveDist - 1) * 0.001
    #    0.5 - 0.098
    #    0.402
    if stateValue == 0.402:
        print "SeeSaw - Unit Test #1 Passed"
    else:
        print "[UT - MOVE_ANT_VALUE] Failure"        
else:
    print "[UT - MOVE_ANT] Failure"


















