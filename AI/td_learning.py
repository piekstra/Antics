import math, heapq, sys, pickle
import os.path as filePath
from Player import *
from Construction import Construction
from Ant import Ant
from AIPlayerUtils import *

#file name constant
UTILITY_FILE = "underwoo16_piekstra17_utility.p"

# minimax constants

INFINITY = 9999.9
MAX_DEPTH = 1
SET_POOL = 4

# neural network constants

NUM_INPUTS = 8
NUM_NODES = 2 * NUM_INPUTS
LEARNING_RATE = 0.1
DUMP_TRIGGER = 10000

# True: agent loads network weights from file "weights.p" and plays only using neural network output
# False: agent uses heuristic evaluation function and back propagation to find best network weights
LOAD_WEIGHTS_FROM_FILE = True

# a representation of a 'node' in the search tree
treeNode = {
    # backreference to parent node
    "parent"            : None,
    # the Move that would be taken in the given state from the parent node
    "move"              : None,
    # the state that would be reached by taking the above move
    "potential_state"   : None,
    # an evaluation of the potential_state
    "state_value"       : 0.0,
    # the utility of the parent state (default super negative)
    "state_utility"       : -1000.0,
    # alpha/beta values for minimax
    "alpha"             : -INFINITY,
    "beta"              : INFINITY,
}

##
# AIPlayer
#
# Description: The responsibility of this class is to interact with the game by
#   deciding a valid move based on a given game state. This class has methods that
#   will be implemented by students in Dr. Nuxoll's AI course.
#
# Variables:
#    playerId - The id of the player.
##
class AIPlayer(Player):

    ##
    # __init__
    #
    # Description: Creates a new AI Player that uses the Minimax algorithm
    #   and maintains a neural network.
    #
    # Parameters:
    #   inputPlayerId - The id to give the new player (int)
    ##
    def __init__(self, inputPlayerId):
        # initialize the AI in the game
        super(AIPlayer,self).__init__(inputPlayerId, "TD")

        # reference to AIPlayer's anthill structure
        self.playerAnthill = None

        # initialize our food location tracker to be empty
        self.foodList = []
        
        # initialize the dictionary that will map state tuples to
        # their utilities
        self.utilityDict = {}
        
        #  discount factor for the TD-Learning algorithm
        self.td_lambda = 0.8
        
        # the learning rate for the TD-Learning algorithm
        self.td_alpha = 0.99
        
        # whether or not the first move of the game has been made
        self.firstMove = True
        
        if filePath.isfile(UTILITY_FILE):
            self.utilityDict = self.loadUtilityList()

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
        
        #initialize the compressed state dictionary with default values
        compressedState = {"queen":
                                {"exists": False,
                                 "on_hill": True},
                           "worker1":
                                {"exists": False,
                                "distance": 20},
                           "worker2":
                                {"exists": False,
                                "distance": 20},
                            "enemy_queen":
                                {"exists": True,
                                 "health": 4}}
        
        
        #keep track of the number of workers for dictionary reference
        workerCount = 0

        # check each of the player's ants
        for ant in playerInv.ants:
            if ant.type == QUEEN:
                # keep track of if the queen exists and is on hill
                compressedState["queen"]["exists"] = True
                
                #whether or not the queen is on the hill
                if ant.coords == self.playerAnthill:
                    compressedState["queen"]["on_hill"] = True
                else:
                    compressedState["queen"]["on_hill"] = False
            
            #for each worker
            elif ant.type == WORKER:
            
                #set the existence variable to true
                workerCount += 1
                    
                workerID = "worker" + str(workerCount)
                compressedState[workerID]["exists"] = True
                
                #store the distance from the worker to enemy queen
                if enemyQueen:
                    compressedState[workerID]["distance"] = (abs(ant.coords[0] - enemyQueen.coords[0]) +
                                                          abs(ant.coords[1] - enemyQueen.coords[1]))
                else:
                    #if enemy queen does not exist, default to zero
                    compressedState[workerID]["distance"] = 0

        if enemyQueen:
            #keep track of existence of enemy queen
            compressedState["enemy_queen"]["exists"] = True
            # keep track of health of enemy queen
            compressedState["enemy_queen"]["health"] = enemyQueen.health
        else:
            #keep track of existence of enemy queen
            compressedState["enemy_queen"]["exists"] = False
            # keep track of health of enemy queen
            compressedState["enemy_queen"]["health"] = 0
                  
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
        if not compressedState["enemy_queen"]["exists"]:
            return 1.0
        if not compressedState["queen"]["exists"] or (not compressedState["worker1"]["exists"] and not compressedState["worker2"]["exists"]) :
            return -1.0
        return -0.01 * compressedState["enemy_queen"]["health"]
            
            
    ##
    # dictToTuple
    #
    # Description: Converts a consolidatedState dictionary to a tuple
    #
    # Parameters:
    #   stateDict - the consolidated state to convert'
    #
    # Returns:
    #   the corresponding tuple representation
    ##
    def dictToTuple(self, stateDict):
        stateArray = []
        for item in stateDict.items():
            stateArray.append((item[0], tuple(item[1].items())))
        return tuple(stateArray)
        
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
         
            
    ##
    # evaluateNodes
    #
    # Description: Takes a list of nodes and returns the min or
    # max node depending on whose turn it is
    #
    # Parameters:
    #   nodes - The list of nodes to evaluate
    #
    # Return: The appropriate min or max node
    ##
    def evaluateNodes(self, nodes):
        # look through the nodes and find the best min/max state value
        # based on the appropriate alpha/beta score and whose turn it is
        if self.playerId == nodes[0]['potential_state'].whoseTurn:
            # MAX node
            maxScore = max(n['alpha'] for n in nodes)
            bestNodes = [n for n in nodes if n['alpha'] == maxScore]
            return random.choice(bestNodes)
        else:
            # MIN node
            minScore = min(n['beta'] for n in nodes)
            bestNodes = [n for n in nodes if n['beta'] == minScore]
            return random.choice(bestNodes)
    
    ##
    # exploreTree
    #
    # Description: Explores the move search tree recursively and
    #   returns the node with the Move that leads to the best branch
    #   in the tree using alpha/beta pruning
    #
    # Parameters:
    #   currentNode - The State being 'searched' from
    #   currentDepth - The depth of the recursive tree search
    #
    # Return: A node containing the best predicted Move
    ##
    def exploreTree(self, currentNode, currentDepth):

        # create alias lookup/reference variables
        currentState = currentNode["potential_state"]
        whoseTurn = currentState.whoseTurn

        # determine whether this is a Min or a Max player
        isMax = whoseTurn == self.playerId

        # generate a list of Nodes from the possible Moves
        possibleNodes = []
        for m in listAllLegalMoves(currentState):

            #ignore moving the queen if she is not on structure
            if m.moveType == MOVE_ANT:
                if getAntAt(currentState, m.coordList[0]) == QUEEN:
                    if getConstrAt(currentState, m.coordList[0]) is None:
                        continue
                        
            # ignore any build moves if the player already has 3 ants (queen and two others)            
            if m.moveType == BUILD:
                if len(currentState.inventories[currentState.whoseTurn].ants) >= 3:
                    continue

            # initialize the node properties
            node = treeNode.copy()
            node["parent"] = currentNode
            node["move"] = m
            node["potential_state"] = self.getFutureState(currentState, m)
            #use the heuristic function
            node["state_value"] = self.evaluateState(node["potential_state"])
            # comments!
            stateUtil = self.addStateUtility(currentState, node["potential_state"])
            node["state_utility"] = stateUtil
            possibleNodes.append(node)

        # intelligently select a subset of nodes to expand
        nodesToIterate = heapq.nlargest(SET_POOL, possibleNodes, key=lambda x: x["state_value"])

        # expand and evaluate the subnodes for this node
        for newNode in nodesToIterate:

            # BASE STEP (leaf node)
            if currentDepth == MAX_DEPTH:


                #use the heuristic function
                nodeVal = self.evaluateState(newNode["potential_state"])

                # set alpha/beta based whether MAX or MIN node
                if isMax:
                    newNode["alpha"] = nodeVal
                else:
                    newNode["alpha"] = 0 - nodeVal
                newNode["beta"] = newNode["alpha"]

                # flip whoseTurn if this an END move and reset the ants so that they can move
                # (this must be done after evaluateState so that evaluateState knows whose turn it is)
                if newNode["move"].moveType == END:
                    newNode["potential_state"].whoseTurn = 1 - newNode["potential_state"].whoseTurn
                    for ant in newNode["potential_state"].inventories[newNode["potential_state"].whoseTurn].ants:
                        ant.hasMoved = False

            # RECURSIVE STEP (branch node)
            else:
                # get a back-reference to the parent node
                parentNode = currentNode["parent"]

                # do alpha/beta pruning if we're not the root node
                if parentNode:
                    # parent is a MAX and current node is a MIN
                    if parentNode["potential_state"].whoseTurn == self.playerId:
                        if (not isMax) and parentNode["alpha"] > currentNode["beta"]:
                            break
                    # parent is a MIN and current node is a MAX
                    elif isMax and parentNode["beta"] < currentNode["alpha"]:
                            break

                # flip whoseTurn if this an END move and reset the ants so that they can move
                # (this must be done after evaluateState so that evaluateState knows whose turn it is)
                if newNode["move"].moveType == END:
                    newNode["potential_state"].whoseTurn = 1 - newNode["potential_state"].whoseTurn
                    for ant in newNode["potential_state"].inventories[newNode["potential_state"].whoseTurn].ants:
                        ant.hasMoved = False

                # recurse on the subnode
                self.exploreTree(newNode, currentDepth + 1)

            # update our alpha/beta values based on our expanded subnode's values
            if isMax:
                # current node is a MAX node
                if newNode["potential_state"].whoseTurn != self.playerId:
                    # subnode is a MIN node
                    if newNode["beta"] > currentNode["alpha"]:
                        currentNode["alpha"] = newNode["beta"]
                else:
                    # subnode is a MAX node
                    if newNode["alpha"] > currentNode["alpha"]:
                        currentNode["alpha"] = newNode["alpha"]
            else:
                # current node is a MIN node
                if newNode["potential_state"].whoseTurn == self.playerId:
                    # subnode is a MAX node
                    if newNode["alpha"] < currentNode["beta"]:
                        currentNode["beta"] = newNode["alpha"]
                else:
                    # subnode is a MIN node
                    if newNode["beta"] < currentNode["beta"]:
                        currentNode["beta"] = newNode["beta"]

        # return our best subnode
        # return self.evaluateNodes(nodesToIterate)
        # TODO COMMENTS
        # return the node with the highest utility
        return sorted(nodesToIterate, key=lambda node: node['state_utility'])[-1]

    ##
    # getFutureState
    #
    # Description: Simulates and returns the new state that would exist after
    #   a move is applied to current state
    #
    # Parameters:
    #   currentState - The game's current state (GameState)
    #   move - The Move that is to be simulated
    #
    # Return: The simulated future state (GameState)
    ##
    def getFutureState(self, currentState, move):
        # create a bare-bones copy of the state to modify
        newState = currentState.fastclone()

        # get references to the player inventories
        playerInv = newState.inventories[newState.whoseTurn]
        enemyInv = newState.inventories[1 - newState.whoseTurn]

        if move.moveType == BUILD:
        # BUILD MOVE
            if move.buildType < 0:
                # building a construction
                playerInv.foodCount -= CONSTR_STATS[move.buildType][BUILD_COST]
                playerInv.constrs.append(Construction(move.coordList[0], move.buildType))
            else:
                # building an ant
                playerInv.foodCount -= UNIT_STATS[move.buildType][COST]
                playerInv.ants.append(Ant(move.coordList[0], move.buildType, newState.whoseTurn))

        elif move.moveType == MOVE_ANT:
        # MOVE AN ANT
            # get a reference to the ant
            ant = getAntAt(newState, move.coordList[0])

            # update the ant's location after the move
            ant.coords = move.coordList[-1]
            ant.hasMoved = True

            # get a reference to a potential construction at the destination coords
            constr = getConstrAt(newState, move.coordList[-1])

            # check to see if a worker ant is on a food or tunnel or hill and act accordingly
            if constr and ant.type == WORKER:
                # if destination is food and ant can carry, pick up food
                if constr.type == FOOD:
                    if not ant.carrying:
                        ant.carrying = True
                # if destination is dropoff structure and and is carrying, drop off food
                elif constr.type == TUNNEL or constr.type == ANTHILL:
                    if ant.carrying:
                        ant.carrying = False
                        playerInv.foodCount += 1

            # get a list of the coordinates of the enemy's ants
            enemyAntCoords = [enemyAnt.coords for enemyAnt in enemyInv.ants]

            # contains the coordinates of ants that the 'moving' ant can attack
            validAttacks = []

            # go through the list of enemy ant locations and check if
            # we can attack that spot, and if so add it to a list of
            # valid attacks (one of which will be chosen at random)
            for coord in enemyAntCoords:
                if UNIT_STATS[ant.type][RANGE] ** 2 >= abs(ant.coords[0] - coord[0]) ** 2 + \
                                                       abs(ant.coords[1] - coord[1]) ** 2:
                    validAttacks.append(coord)

            # if we can attack, pick a random attack and do it
            if validAttacks:
                enemyAnt = getAntAt(newState, random.choice(validAttacks))
                attackStrength = UNIT_STATS[ant.type][ATTACK]

                if enemyAnt.health <= attackStrength:
                    # just to be safe, set the health to 0
                    enemyAnt.health = 0
                    # remove the enemy ant from their inventory
                    enemyInv.ants.remove(enemyAnt)
                else:
                    # lower the enemy ant's health because they were attacked
                    enemyAnt.health -= attackStrength

        # return the modified copy of the original state
        return newState

    ##
    # evaluateState
    #
    # Description: The evaluateState method looks at a state and
    #   assigns a value to the state based on how well the game is
    #   going for the current player
    #
    # Parameters:
    #   currentState - The State to evaluate
    #
    # Return: The score of the state on a scale of 0.0 to 1.0
    #   where 0.0 is a loss and 1.0 is a victory and 0.5 is neutral
    ##
    def evaluateState(self, state):
        # local constants for adjusting weights of evaluation
        QUEEN_EDGE_MAP_WEIGHT = 0.01
        BUILD_WORKER_WEIGHT = 0.1
        MOVE_TOWARDS_QUEEN_WEIGHT = 0.005
        QUEEN_HEALTH_WEIGHT = 0.025
        QUEEN_ON_HILL_WEIGHT = 0.1

        # get some references for use later
        playerInv = state.inventories[state.whoseTurn]
        enemyInv = state.inventories[1 - state.whoseTurn]
        enemyQueen = enemyInv.getQueen()

        # start the score at a semi-neutral value
        score = 0.4

        # check each of the player's ants
        for ant in playerInv.ants:
            if ant.type == QUEEN:
                # keep the queen off of the anthill
                if ant.coords == self.playerAnthill:
                    score -= QUEEN_ON_HILL_WEIGHT
                # keep the queen at the edge of the map
                score -= QUEEN_EDGE_MAP_WEIGHT * ant.coords[1]
            elif ant.type == WORKER:
                # encourage building more workers
                score += BUILD_WORKER_WEIGHT
                if enemyQueen:
                    # move workers towards queen
                    score -= MOVE_TOWARDS_QUEEN_WEIGHT * (abs(ant.coords[0] - enemyQueen.coords[0]) +
                                                          abs(ant.coords[1] - enemyQueen.coords[1]))

        if enemyQueen:
            # encourage damaging the enemy queen
            score += QUEEN_HEALTH_WEIGHT * (UNIT_STATS[QUEEN][HEALTH] - enemyQueen.health)
        else:
            score += QUEEN_HEALTH_WEIGHT * UNIT_STATS[QUEEN][HEALTH]

        # return the evaluation score of the state
        return score
           
    ##
    # getPlacement
    #
    # Description: called twice during setup phase to place objects that
    #   must be placed by the player.  These items are: 1 Anthill on
    #   the player's side; 1 tunnel on player's side; 9 grass on the
    #   player's side; and 2 food on the enemy's side.
    #
    # Parameters:
    #   currentState - the state of the game at this point in time.
    #
    # Return: A list of coordinates of where the constructions are to be placed
    ##
    def getPlacement(self, currentState):
        self.foodList = []
        #implemented by students to return their next move
        if currentState.phase == SETUP_PHASE_1:    #stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move is None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr is None and (x, y) not in moves:
                        move = (x, y)
                        if i == 0:
                            self.playerAnthill = move
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:   #stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move is None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr is None and (x, y) not in moves:
                        move = (x, y)
                moves.append(move)
            return moves
        else:
            return [(0, 0)]
    
    
    ## TODO COMMENTS
    def addStateUtility(self, currentState, nextState=None):     
        # convert the current state to tuple form to be used as a dictionary key
        curStateDict = self.consolidateState(currentState)
        curStateTuple = self.dictToTuple(curStateDict) 
        
        # in the case of the first move of a game, if we have not seen
        # the initial state, set it's utility to 0, otherwise leave it
        if nextState is None:
            if curStateTuple not in self.utilityDict:
                # state not previously encountered, default utility to 0
                self.utilityDict[curStateTuple] = 0
        else:    
            # convert the next state to tuple form to be used as a dictionary key
            nextStateTuple = self.dictToTuple(self.consolidateState(nextState))     
            # store the state's utility in the dict
            if nextStateTuple not in self.utilityDict:
                # state not previously encountered, default utility to 0
                self.utilityDict[nextStateTuple] = 0
            else:
                self.utilityDict[curStateTuple] += self.td_alpha*(self.rewardFunction(curStateDict) + self.td_lambda*self.utilityDict[nextStateTuple] - self.utilityDict[curStateTuple])
        print self.utilityDict[curStateTuple], curStateTuple
        return self.utilityDict[curStateTuple]
    
    ##
    # getMove
    #
    # Description: Gets the next move from the Player.
    #
    # Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    # Return: The Move to be made
    ##
    def getMove(self, currentState): 
        # on the first move, add the starting state to the utility dict
        # if it isn't already there
        if self.firstMove:
            self.firstMove = False
            self.addStateUtility(currentState)
            
        #fill the food locations list
        if not self.foodList:
            for f in getConstrList(currentState, NEUTRAL, (FOOD,)):
                self.foodList.append(f)

        # create a root node to build our search tree from
        rootNode = treeNode.copy()
        rootNode["potential_state"] = currentState
        
        
        # # convert the state to tuple form so it can be used as a dictionary key
        # stateTuple = self.dictToTuple(self.consolidateState(currentState))     
        # # store the state's utility in the dict
        # if stateTuple not in self.utilityDict:
            # # state not previously encountered, default utility to 0
            # self.utilityDict[stateTuple] = 0
        # else:
            
            
        # return the best move, found by recursively searching potential moves
        return self.exploreTree(rootNode, 0)["move"]

    ##
    # getAttack
    #
    # Description: Gets the attack to be made from the Player
    #
    # Parameters:
    #   currentState - A clone of the current state (GameState)
    #   attackingAnt - The ant currently making the attack (Ant)
    #   enemyLocation - The Locations of the Enemies that can be attacked (Location[])
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        # attack a random enemy
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]
        
    ##
    # registerWin
    #
    # Description: Tells the player if they won or not
    #
    # Parameters:
    #   hasWon - True if the player won the game. False if they lost (Boolean)
    #
    def registerWin(self, hasWon):
        # set the first move back to True for the next game
        self.firstMove = True
        # TODO COMMENTS
        x = self.td_alpha
        decreaseBy = (1 - (1/(math.e**x**8)))*0.1
        self.td_alpha -= decreaseBy
        print "LR-B4: %d\tLR-AF: %d\tDelta: %d" % (x, self.td_alpha, decreaseBy)
        self.saveUtilityList()
