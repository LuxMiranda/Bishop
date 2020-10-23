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
    # If more than one object satisfy the criterion, there is an indeterminate
    # priority between then.
    if len(goals) > 1:
        return -1
    # Likewise if no objects satisfy it.
    if len(goals) == 0:
        return -1
    # Else, return the one object
    return goals[0]


def getGoal(gameMap, player, actions):
    """
    Compute the most likely goal of the given player and action sequence
    """
    # Build observer and run inference
    Observer = LoadObserver("{}_Player{}".format(gameMap,player), Silent=True)
    Results = Observer.InferAgent(
        ActionSequence=actions, Samples=50, Feedback=False)
    # Fetch object names and inferred reward matrix
    objects = Results.ObjectNames
    rewardMatrix = Results.CompareRewards()
    # Compute the most likely goal object
    likelyGoal = mostLikelyGoalObject(rewardMatrix)
    # Goal object of -1 is a sentinel value for an indeterminate result
    goal = objects[likelyGoal] if likelyGoal != -1 else 'Indeterminate'
    return goal

def cooperating(goal1, goal2):
    return (goal1 == goal2) and (goal1 != 'Indeterminate')# and (goal1[:-1] == 'Stag')

def inferCooperators(gameMap, actions):
    """
    Given a game map and action set for each player, infer who is cooperating.
    """
    goalA = getGoal(gameMap, 'A', actions['A'])
    goalB = getGoal(gameMap, 'B', actions['B'])
    goalC = getGoal(gameMap, 'C', actions['C'])
    cooperators = {
        'AB' : cooperating(goalA, goalB),
        'AC' : cooperating(goalA, goalC),
        'BC' : cooperating(goalB, goalC)
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
    print(gameMap)
    actions = {
        'A' : ['R'],
        'B' : ['U'],
        'C' : ['L'],
    }
    inferredCoops = inferCooperators(gameMap, actions)
    print(inferredCoops)
    correctPredictions += numCorrects(trueCoops, inferredCoops)

    ### Scenario (b)
    trueCoops = {'AC': False, 'AB': False, 'BC': False}
    gameMap = 'StagHunt_b_T012'
    print(gameMap)
    actions = {
        'A' : ['U'],
        'B' : ['R'],
        'C' : ['D'],
    }
    inferredCoops = inferCooperators(gameMap, actions)
    print(inferredCoops)
    correctPredictions += numCorrects(trueCoops, inferredCoops)

    ### Scenario (c)
    trueCoops = {'AC': False, 'AB': False, 'BC': True}
    gameMap = 'StagHunt_c_T01'
    print(gameMap)
    actions = {
        'A' : ['L'],
        'B' : ['R'],
        'C' : ['D'],
    }
    inferredCoops = inferCooperators(gameMap, actions)
    print(inferredCoops)
    correctPredictions += numCorrects(trueCoops, inferredCoops)

    ### Scenario (d)
    trueCoops = {'AC': False, 'AB': True, 'BC': False}
    gameMap = 'StagHunt_d'
    print(gameMap)
    actions = {
        'A' : ['U'],
        'B' : ['U'],
        'C' : ['U'],
    }
    inferredCoops = inferCooperators(gameMap, actions)
    print(inferredCoops)
    correctPredictions += numCorrects(trueCoops, inferredCoops)

    ### Scenario (e)
    trueCoops = {'AC': False, 'AB': False, 'BC': False}
    gameMap = 'StagHunt_e'
    print(gameMap)
    actions = {
        'A' : ['L'],
        'B' : ['D'],
        'C' : ['R'],
    }
    inferredCoops = inferCooperators(gameMap, actions)
    print(inferredCoops)
    correctPredictions += numCorrects(trueCoops, inferredCoops)

    ### Scenario (f)
    trueCoops = {'AC': False, 'AB': False, 'BC': False}
    gameMap = 'StagHunt_f'
    print(gameMap)
    actions = {
        'A' : ['R'],
        'B' : ['D'],
        'C' : [],
    }
    inferredCoops = inferCooperators(gameMap, actions)
    print(inferredCoops)
    correctPredictions += numCorrects(trueCoops, inferredCoops)

    ### Scenario (g)
    trueCoops = {'AC': True, 'AB': True, 'BC': True}
    gameMap = 'StagHunt_g_T01'
    print(gameMap)
    actions = {
        'A' : ['R'],
        'B' : ['R'],
        'C' : ['R'],
    }
    inferredCoops = inferCooperators(gameMap, actions)
    print(inferredCoops)
    correctPredictions += numCorrects(trueCoops, inferredCoops)

    ### Scenario (h)
    trueCoops = {'AC': False, 'AB': False, 'BC': False}
    gameMap = 'StagHunt_h_T01'
    print(gameMap)
    actions = {
        'A' : ['U'],
        'B' : ['U'],
        'C' : ['R'],
    }
    inferredCoops = inferCooperators(gameMap, actions)
    print(inferredCoops)
    correctPredictions += numCorrects(trueCoops, inferredCoops)

    ### Scenario (i)
    trueCoops = {'AC': True, 'AB': True, 'BC': True}
    gameMap = 'StagHunt_i_T01'
    print(gameMap)
    actions = {
        'A' : ['L'],
        'B' : ['L'],
        'C' : ['D'],
    }
    inferredCoops = inferCooperators(gameMap, actions)
    print(inferredCoops)
    correctPredictions += numCorrects(trueCoops, inferredCoops)

    ### Tally up
    totalPredictions = 3.0 * 9
    accuracy = correctPredictions/totalPredictions
    print(accuracy)

if __name__ == '__main__':
    main()

#print("PREDICTING NEXT ACTION(S)")
#Probs = Observer.PredictAction(Results)
#for plan,prob in list(zip(Probs[0],Probs[1])):
#   print(plan,prob)
