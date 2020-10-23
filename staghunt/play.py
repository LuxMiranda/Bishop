from Bishop import *
import numpy as np

def mostLikelyGoalObject(rewardMatrix):
    """
    Return the object index of an object that is greater than 50% probable
    to have a greater reward than all other objects.
    """
    # Given a list of values, return True if every value is greater than 0.5
    allLikely = lambda ls : False if False in [x>0.5 for x in ls] else True
    # Apply function to all objects
    goals = []
    for i in range(4):
        if allLikely(rewardMatrix[i]):
            goals.append(i)
    # In theory, only one object should satisfy the criterion. But double check
    # to be safe.
    if len(goals) > 1:
        raise SystemExit('Multiple objects satisfy probable goal criterion.')
    # If no object satisfied the criterion, list is empty. Return sentinel value.
    if len(goals) == 0:
        return -1
    # Else, return the one object
    return goals[0]


def getGoal(gameMap, player, actions):
    """
    Compute the most likely goal of the given player and action sequence
    """
    # Build observer and run inference
    Observer = LoadObserver("StagHunt_a_Player{}".format(player))
    Results = Observer.InferAgent(
        ActionSequence=actions, Samples=5, Feedback=False)
    # Fetch object names and inferred reward matrix
    objects = Results.ObjectNames
    rewardMatrix = Results.CompareRewards()
    # Compute the most likely goal object
    likelyGoal = mostLikelyGoalObject(rewardMatrix)
    # Goal object of -1 is a sentinel value for an indeterminate result
    goal = objects[likelyGoal] if likelyGoal != -1 else 'Indeterminate'
    return goal

def inferCooperators(gameMap, actions):
    """
    Given a game map and action set for each player, infer who is cooperating.
    """
    goalA = getGoal(gameMap, 'A', actions['A'])
    goalB = getGoal(gameMap, 'B', actions['B'])
    goalC = getGoal(gameMap, 'C', actions['C'])
    cooperators = {
        'AB' : (goalA == goalB and goalA != 'Indeterminate'),
        'AC' : (goalA == goalC and goalB != 'Indeterminate'),
        'BC' : (goalB == goalC and goalB != 'Indeterminate')
    }
    return cooperators

def numCorrects(trueCoops, inferredCoops):
    """
    Count the number of correct predictions given the true and inferred
    cooperator pairs.
    """
    numCorrects = 0
    if trueCoops['AC'] == inferredCoops['AC']:
        numCorrects += 1
    if trueCoops['AB'] == inferredCoops['AB']:
        numCorrects += 1
    if trueCoops['BC'] == inferredCoops['BC']:
        numCorrects += 1
    return numCorrects

def main():

    ###
    ### TIMESTEP 1
    ###
    correctPredictions = 0.0

    ### Scenario (a)
    trueCoops = {'AC': True, 'AB': False, 'BC': False}
    gameMap = 'StagHunt_a'
    actions = {
        'A' : ['R'],
        'B' : ['U'],
        'C' : ['L'],
    }
    inferredCoops = inferCooperators(gameMap, actions)
    print(inferredCoops)
    correctPredictions += numCorrects(trueCoops, inferredCoops)

    ### Tally up
    totalPredictions = 3.0
    accuracy = correctPredictions/totalPredictions
    print(accuracy)

if __name__ == '__main__':
    main()

#print("PREDICTING NEXT ACTION(S)")
#Probs = Observer.PredictAction(Results)
#for plan,prob in list(zip(Probs[0],Probs[1])):
#   print(plan,prob)
