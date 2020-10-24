from Bishop import *
import numpy as np
from multiprocessing import Pool
import time
import logging as log
from tqdm import tqdm

# Set start time to the (rough) moment the program starts up
START_TIME = time.strftime('%Y%m%d-%H%M%S')

# Set up logging
log.basicConfig(
    filename='log/{}.log'.format(START_TIME),
    filemode='w',
    format='[%(asctime)s - %(levelname)s] %(message)s',
    level=log.INFO
)

# More samples = more better = more longer
SAMPLES_PER_INFERENCE = 5

# Experiment isn't deterministic (Bishop relies on a few random numbers), so
# this controls the number of times to repeat the full experiment to acquire an
# average accuracy across all runs.
TIMESTEP_REPEATS = 1

def mostLikelyGoalObject(rewardMatrix):
    """
    Return the object index of an object that is at least >50% probable
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
    # priority between them.
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
        ActionSequence=actions, Samples=SAMPLES_PER_INFERENCE, Feedback=False)
    # Fetch object names and inferred reward matrix
    objects = Results.ObjectNames
    rewardMatrix = Results.CompareRewards()
    # Compute the most likely goal object
    likelyGoal = mostLikelyGoalObject(rewardMatrix)
    # Goal object of -1 is a sentinel value for an indeterminate result
    goal = objects[likelyGoal] if likelyGoal != -1 else 'Indeterminate'
    # Log function ret
    log.info('getGoal(gameMap={},player={},actions={}) returned {}'.format(
        gameMap,player,actions,goal
    ))
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


def runScenario(args):
    """
    Use scenario parameters to make a cooperation inference. Returns the 
    number of correct predictions (min 0, max 3).
    """
    # Unpack args
    trueCoops, gameMap, actions = args['trueCoops'],args['gameMap'],args['actions']
    # Infer cooperators
    inferredCoops = inferCooperators(gameMap, actions)
    # Log 
    log.info('inferCooperators() on {} finish: trueCoops={}, inferredCoops={}'.format(
        gameMap, trueCoops, inferredCoops
    ))
    # Return number of correct predictions
    return numCorrects(trueCoops, inferredCoops)

def timestep1(i):
    ###
    ### TIMESTEP 1
    ###
    scenarios = [
        {### Scenario (a)
            'trueCoops' : {'AC': True, 'AB': False, 'BC': False},
            'gameMap'   : 'StagHunt_a',
            'actions'   : { 'A' : ['R'], 'B' : ['U'], 'C' : ['L'] }
        },
        {### Scenario (b)
            'trueCoops' : {'AC': False, 'AB': False, 'BC': False},
            'gameMap'   : 'StagHunt_b_T012',
            'actions'   : { 'A' : ['U'], 'B' : ['R'], 'C' : ['D'] }
        },
        {### Scenario (c)
            'trueCoops' : {'AC': False, 'AB': False, 'BC': True},
            'gameMap'   : 'StagHunt_c_T01',
            'actions'   : { 'A' : ['L'], 'B' : ['R'], 'C' : ['D'] }
        },
        {### Scenario (d)
            'trueCoops' : {'AC': False, 'AB': True, 'BC': False},
            'gameMap'   : 'StagHunt_d',
            'actions'   : { 'A' : ['U'], 'B' : ['U'], 'C' : ['U'] }
        },
        {### Scenario (e)
            'trueCoops' : {'AC': False, 'AB': False, 'BC': False},
            'gameMap'   : 'StagHunt_e',
            'actions'   : { 'A' : ['L'], 'B' : ['D'], 'C' : ['R'] }
        },
        {### Scenario (f)
            'trueCoops' : {'AC': False, 'AB': False, 'BC': False},
            'gameMap'   : 'StagHunt_f',
            'actions'   : { 'A' : ['R'], 'B' : ['D'], 'C' : [] }
        },
        {### Scenario (g)
            'trueCoops' : {'AC': True, 'AB': True, 'BC': True},
            'gameMap'   : 'StagHunt_g_T01',
            'actions'   : { 'A' : ['R'], 'B' : ['R'], 'C' : ['R'] }
        },
        {### Scenario (h)
            'trueCoops' : {'AC': False, 'AB': False, 'BC': False},
            'gameMap'   : 'StagHunt_h_T01',
            'actions'   : { 'A' : ['U'], 'B' : ['U'], 'C' : ['R'] }
        },
        {### Scenario (i)
            'trueCoops' : {'AC': True, 'AB': True, 'BC': True},
            'gameMap'   : 'StagHunt_i_T01',
            'actions'   : { 'A' : ['L'], 'B' : ['L'], 'C' : ['D'] }
        },
    ]
    
    correctPredictions = np.sum([runScenario(s) for s in scenarios])
    totalPredictions = 3.0 * 9
    accuracy = correctPredictions/totalPredictions
    log.info('### TIMESTEP 1 RUN {} FINISHED. Accuracy: {}'.format(i,accuracy))
    return accuracy

def runExperiment():
    # Note to user: Adjust thread count proportionally to hardware chonkiness
    numThreads = 2
    pool = Pool(numThreads)
    log.info('Beginning experiment with {} samples per inference across {}\
            timestep repeats running on {} threads.'.format(
                SAMPLES_PER_INFERENCE,
                TIMESTEP_REPEATS,
                numThreads
            ))
    accuracies = list(
            tqdm(
                pool.imap(timestep1, list(range(TIMESTEP_REPEATS))), 
                total=TIMESTEP_REPEATS
                )
            )
    log.critical('### EXPERIMENT COMPLETE ###')
    log.critical('Accuracies: {}'.format(accuracies))
    log.critical('Mean accuracy: {}'.format(np.mean(accuracies)))

    filename='log/{}.log'.format(START_TIME)
    print('Logged run to {}'.format(filename))

def main():
    print(timestep1(0))

if __name__ == '__main__':
    main()

#print("PREDICTING NEXT ACTION(S)")
#Probs = Observer.PredictAction(Results)
#for plan,prob in list(zip(Probs[0],Probs[1])):
#   print(plan,prob)
