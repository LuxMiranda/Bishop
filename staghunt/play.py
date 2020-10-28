from Bishop import *
import numpy as np
from multiprocessing import Pool
import time
import logging
from tqdm import tqdm

START_TIME = time.strftime('%Y%m%d-%H%M%S')

# From https://stackoverflow.com/a/31695996
def setup_logger(
        logger_name, 
        log_file, format='[%(asctime)s - %(levelname)s] %(message)s',
        level=logging.INFO ):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(format)
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fileHandler)

setup_logger('timestep1','log/timestep1-{}.log'.format(START_TIME))
setup_logger('timestep2','log/timestep2-{}.log'.format(START_TIME))
setup_logger('timestep3','log/timestep3-{}.log'.format(START_TIME)) 
setup_logger('actionpred_timestep1','log/actionpred1-{}.csv'.format(START_TIME), format='%(message)s')
setup_logger('actionpred_timestep2','log/actionpred2-{}.csv'.format(START_TIME), format='%(message)s')

# More samples = more better = more longer
SAMPLES_PER_INFERENCE = 500
#SAMPLES_PER_INFERENCE = 10

# Experiment isn't deterministic (Bishop relies on a few random numbers), so
# this controls the number of times to repeat the full experiment to acquire an
# average accuracy across all runs.
TIMESTEP_REPEATS = 50

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

def predictNextAction(observer, results):
    # Fetch the likelihoods for each action
    # [L, R, U, D]
    likelihoods = observer.PredictAction(results)[1]
    # Find the maximum likelihood
    maxProb = np.max(likelihoods)
    # If two or more likelihood are equal to the max
    if len([p for p in likelihoods if p == maxProb]) >= 2:
        # Indecision; return no movement
        return []
    # Otherwise, return the single action with the highest probability
    return [['L'],['R'],['U'],['D']][likelihoods.index(maxProb)]

def predictActions(gameMap, player, actions, timestep, observer, results, trueActions, goal):
    if timestep == 'timestep3':
        return
    # Predict the next agent action
    nextAction = predictNextAction(observer, results)
    predActions = nextAction[0] if nextAction != [] else 'N'
    # If it's timestep 1
    if timestep == 'timestep1':
        # Use this action prediction to predict an additional action
        newResults = observer.InferAgent(
                ActionSequence=actions + nextAction,
                Samples=SAMPLES_PER_INFERENCE,
                Feedback=False)
        nextAction = predictNextAction(observer, newResults)
        predActions += nextAction[0] if nextAction != [] else 'N'
    # Log action predictions
    log = logging.getLogger('actionpred_{}'.format(timestep))
    # gameMap, player, reference_timestep, goal, pred_actions, true_actions
    log.info('{},{},{},{},{},{}'.format(
        gameMap, player, timestep, goal, predActions, ''.join(trueActions[player])
    ))
    # No return value
    return

def getGoal(gameMap, player, actions,timestep,trueActions):
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
    log = logging.getLogger(timestep)
    log.info('getGoal(gameMap={},player={},actions={}) returned {}'.format(
        gameMap,player,actions,goal
    ))
    # Predict next actions
    if timestep != 'timestep3':
        predictActions(gameMap, player, actions, timestep, Observer, Results,trueActions, goal)
    return goal

def cooperating(goal1, goal2):
    return (goal1 == goal2) and (goal1 != 'Indeterminate')# and (goal1[:-1] == 'Stag')

def inferCooperators(gameMap, actions, timestep, trueActions):
    """
    Given a game map and action set for each player, infer who is cooperating.
    """
    goalA = getGoal(gameMap, 'A', actions['A'], timestep, trueActions)
    goalB = getGoal(gameMap, 'B', actions['B'], timestep, trueActions)
    goalC = getGoal(gameMap, 'C', actions['C'], timestep, trueActions)
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
    trueActions = []
    # Unpack args
    trueCoops, gameMap, actions, timestep = args['trueCoops'],args['gameMap'],args['actions'],args['timestep']
    if timestep != 'timestep3':
        trueActions = args['trueActions']
    # Infer cooperators
    inferredCoops = inferCooperators(gameMap, actions, timestep, trueActions)
    # Log 
    log = logging.getLogger(timestep)
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
            'trueCoops'  : {'AC': True, 'AB': False, 'BC': False},
            'timestep'   : 'timestep1',
            'gameMap'    : 'StagHunt_a',
            'actions'    : { 'A' : ['R'], 'B' : ['U'], 'C' : ['L'] },
            'trueActions': { 'A' : ['R','R','D'], 'B' : ['U','U','L'], 'C' : ['L','L','U'] }
        },
        {### Scenario (b)
            'trueCoops' : {'AC': False, 'AB': False, 'BC': False},
            'timestep'  : 'timestep1',
            'gameMap'   : 'StagHunt_b_T012',
            'actions'   : { 'A' : ['U'], 'B' : ['R'], 'C' : ['D'] },
            'trueActions'   : { 'A' : ['U','L','L'], 'B' : ['R','R','R'], 'C' : ['D','D','R'] }
        },
        {### Scenario (c)
            'trueCoops' : {'AC': False, 'AB': False, 'BC': True},
            'timestep'  : 'timestep1',
            'gameMap'   : 'StagHunt_c_T01',
            'actions'   : { 'A' : ['L'], 'B' : ['R'], 'C' : ['D'] },
            'trueActions'   : { 'A' : ['L','D','D'], 'B' : ['R','R','R'], 'C' : ['D','D','D'] }
        },
        {### Scenario (d)
            'trueCoops' : {'AC': False, 'AB': True, 'BC': False},
            'timestep'  : 'timestep1',
            'gameMap'   : 'StagHunt_d',
            'actions'   : { 'A' : ['U'], 'B' : ['U'], 'C' : ['U'] },
            'trueActions'   : { 'A' : ['U','R'], 'B' : ['U','U','R'], 'C' : ['U','U','R'] }
        },
        {### Scenario (e)
            'trueCoops' : {'AC': False, 'AB': False, 'BC': False},
            'timestep'  : 'timestep1',
            'gameMap'   : 'StagHunt_e',
            'actions'   : { 'A' : ['L'], 'B' : ['R'], 'C' : ['D'] },
            'trueActions'   : { 'A' : ['L','L','L'], 'B' : ['R','U','U'], 'C' : ['D','L','L'] }
        },
        {### Scenario (f)
            'trueCoops' : {'AC': False, 'AB': False, 'BC': False},
            'timestep'  : 'timestep1',
            'gameMap'   : 'StagHunt_f',
            'actions'   : { 'A' : ['R'], 'B' : ['D'], 'C' : [] },
            'trueActions'   : { 'A' : ['R','U','U'], 'B' : ['D','D','R'], 'C' : [] }
        },
        {### Scenario (g)
            'trueCoops' : {'AC': True, 'AB': True, 'BC': True},
            'timestep'  : 'timestep1',
            'gameMap'   : 'StagHunt_g_T01',
            'actions'   : { 'A' : ['R'], 'B' : ['R'], 'C' : ['R'] },
            'trueActions'   : { 'A' : ['R','R','D'], 'B' : ['R','R','R'], 'C' : ['R','R','U'] }
        },
        {### Scenario (h)
            'trueCoops' : {'AC': False, 'AB': False, 'BC': False},
            'timestep'  : 'timestep1',
            'gameMap'   : 'StagHunt_h_T01',
            'actions'   : { 'A' : ['U'], 'B' : ['U'], 'C' : ['R'] },
            'trueActions'   : { 'A' : ['U','D','L'], 'B' : ['U','U','R'], 'C' : ['R','R','U'] }
        },
        {### Scenario (i)
            'trueCoops' : {'AC': True, 'AB': True, 'BC': True},
            'timestep'  : 'timestep1',
            'gameMap'   : 'StagHunt_i_T01',
            'actions'   : { 'A' : ['L'], 'B' : ['L'], 'C' : ['D'] },
            'trueActions'   : { 'A' : ['L','D','D'], 'B' : ['L','D','L'], 'C' : ['D','D','R'] }
        },
    ]
    
    correctPredictions = np.sum([runScenario(s) for s in scenarios])
    totalPredictions = 3.0 * 9
    accuracy = correctPredictions/totalPredictions

    log = logging.getLogger('timestep1')
    log.info('### TIMESTEP 1 RUN {} FINISHED. Accuracy: {}'.format(i,accuracy))
    return accuracy

def timestep2(i):
    ###
    ### TIMESTEP 2
    ###
    scenarios = [
        {### Scenario (a)
            'trueCoops' : {'AC': True, 'AB': False, 'BC': False},
            'timestep'  : 'timestep2',
            'gameMap'   : 'StagHunt_a',
            'actions'   : { 'A' : ['R','R'], 'B' : ['U','U'], 'C' : ['L','L'] },
            'trueActions': { 'A' : ['R','R','D'], 'B' : ['U','U','L'], 'C' : ['L','L','U'] }
        },
        {### Scenario (b)
            'trueCoops' : {'AC': False, 'AB': False, 'BC': False},
            'timestep'  : 'timestep2',
            'gameMap'   : 'StagHunt_b_T012',
            'actions'   : { 'A' : ['U','L'], 'B' : ['R','R'], 'C' : ['D','D'] },
            'trueActions'   : { 'A' : ['U','L','L'], 'B' : ['R','R','R'], 'C' : ['D','D','R'] }
        },
        {### Scenario (c)
            'trueCoops' : {'AC': False, 'AB': False, 'BC': True},
            'timestep'  : 'timestep2',
            'gameMap'   : 'StagHunt_c_T23',
            'actions'   : { 'A' : ['L','D'], 'B' : ['R','R'], 'C' : ['D','D'] },
            'trueActions'   : { 'A' : ['L','D','D'], 'B' : ['R','R','R'], 'C' : ['D','D','D'] }
        },
        {### Scenario (d)
            'trueCoops' : {'AC': False, 'AB': True, 'BC': False},
            'timestep'  : 'timestep2',
            'gameMap'   : 'StagHunt_d',
            'actions'   : { 'A' : ['U'], 'B' : ['U','U'], 'C' : ['U','U'] },
            'trueActions'   : { 'A' : ['U','R'], 'B' : ['U','U','R'], 'C' : ['U','U','R'] }
        },
        {### Scenario (e)
            'trueCoops' : {'AC': False, 'AB': False, 'BC': False},
            'timestep'  : 'timestep2',
            'gameMap'   : 'StagHunt_e',
            'actions'   : { 'A' : ['L','L'], 'B' : ['R','U'], 'C' : ['D','L'] },
            'trueActions'   : { 'A' : ['L','L','L'], 'B' : ['R','U','U'], 'C' : ['D','L','L'] }
        },
        {### Scenario (f)
            'trueCoops' : {'AC': False, 'AB': False, 'BC': False},
            'timestep'  : 'timestep2',
            'gameMap'   : 'StagHunt_f',
            'actions'   : { 'A' : ['R','U'], 'B' : ['D','D'], 'C' : [] },
            'trueActions'   : { 'A' : ['R','U','U'], 'B' : ['D','D','R'], 'C' : [] }
        },
        {### Scenario (g)
            'trueCoops' : {'AC': True, 'AB': True, 'BC': True},
            'timestep'  : 'timestep2',
            'gameMap'   : 'StagHunt_g_T23',
            'actions'   : { 'A' : ['R','R'], 'B' : ['R','R'], 'C' : ['R','R'] },
            'trueActions'   : { 'A' : ['R','R','D'], 'B' : ['R','R','R'], 'C' : ['R','R','U'] }
        },
        {### Scenario (h)
            'trueCoops' : {'AC': False, 'AB': False, 'BC': False},
            'timestep'  : 'timestep2',
            'gameMap'   : 'StagHunt_h_T2',
            'actions'   : { 'A' : ['U','D'], 'B' : ['U','U'], 'C' : ['R','R'] },
            'trueActions'   : { 'A' : ['U','D','L'], 'B' : ['U','U','R'], 'C' : ['R','R','U'] }
        },
        {### Scenario (i)
            'trueCoops' : {'AC': True, 'AB': True, 'BC': True},
            'timestep'  : 'timestep2',
            'gameMap'   : 'StagHunt_i_T23',
            'actions'   : { 'A' : ['L','D'], 'B' : ['L','D'], 'C' : ['D','D'] },
            'trueActions'   : { 'A' : ['L','D','D'], 'B' : ['L','D','L'], 'C' : ['D','D','R'] }
        },
    ]
    
    correctPredictions = np.sum([runScenario(s) for s in scenarios])
    totalPredictions = 3.0 * 9
    accuracy = correctPredictions/totalPredictions
    log = logging.getLogger('timestep2')
    log.info('### TIMESTEP 2 RUN {} FINISHED. Accuracy: {}'.format(i,accuracy))
    return accuracy

def timestep3(i):
    ###
    ### TIMESTEP 3
    ###
    scenarios = [
        {### Scenario (a)
            'trueCoops' : {'AC': True, 'AB': False, 'BC': False},
            'timestep'  : 'timestep3',
            'gameMap'   : 'StagHunt_a',
            'actions'   : { 'A' : ['R','R','D'], 'B' : ['U','U','L'], 'C' : ['L','L','U'] }
        },
        {### Scenario (b)
            'trueCoops' : {'AC': False, 'AB': False, 'BC': False},
            'timestep'  : 'timestep3',
            'gameMap'   : 'StagHunt_b_T3',
            'actions'   : { 'A' : ['U','L','L'], 'B' : ['R','R','R'], 'C' : ['D','D','R'] }
        },
        {### Scenario (c)
            'trueCoops' : {'AC': False, 'AB': False, 'BC': True},
            'timestep'  : 'timestep3',
            'gameMap'   : 'StagHunt_c_T23',
            'actions'   : { 'A' : ['L','D','D'], 'B' : ['R','R','R'], 'C' : ['D','D','D'] }
        },
        {### Scenario (d)
            'trueCoops' : {'AC': False, 'AB': True, 'BC': False},
            'timestep'  : 'timestep3',
            'gameMap'   : 'StagHunt_d',
            'actions'   : { 'A' : ['U','R'], 'B' : ['U','U','R'], 'C' : ['U','U','R'] }
        },
        {### Scenario (e)
            'trueCoops' : {'AC': False, 'AB': False, 'BC': False},
            'timestep'  : 'timestep3',
            'gameMap'   : 'StagHunt_e',
            'actions'   : { 'A' : ['L','L','L'], 'B' : ['R','U','U'], 'C' : ['D','L','L'] }
        },
        {### Scenario (f)
            'trueCoops' : {'AC': False, 'AB': False, 'BC': False},
            'timestep'  : 'timestep3',
            'gameMap'   : 'StagHunt_f',
            'actions'   : { 'A' : ['R','U','U'], 'B' : ['D','D','R'], 'C' : [] }
        },
        {### Scenario (g)
            'trueCoops' : {'AC': True, 'AB': True, 'BC': True},
            'timestep'  : 'timestep3',
            'gameMap'   : 'StagHunt_g_T23',
            'actions'   : { 'A' : ['R','R','D'], 'B' : ['R','R','R'], 'C' : ['R','R','U'] }
        },
        {### Scenario (h)
            'trueCoops' : {'AC': False, 'AB': False, 'BC': False},
            'timestep'  : 'timestep3',
            'gameMap'   : 'StagHunt_h_T3',
            'actions'   : { 'A' : ['U','D','L'], 'B' : ['U','U','R'], 'C' : ['R','R','U'] }
        },
        {### Scenario (i)
            'trueCoops' : {'AC': True, 'AB': True, 'BC': True},
            'timestep'  : 'timestep3',
            'gameMap'   : 'StagHunt_i_T23',
            'actions'   : { 'A' : ['L','D','D'], 'B' : ['L','D','L'], 'C' : ['D','D','R'] }
        },
    ]
    
    correctPredictions = np.sum([runScenario(s) for s in scenarios])
    totalPredictions = 3.0 * 9
    accuracy = correctPredictions/totalPredictions
    log = logging.getLogger('timestep3')
    log.info('### TIMESTEP 3 RUN {} FINISHED. Accuracy: {}'.format(i,accuracy))
    return accuracy


def runExperiment(timestep):
    # Note to user: Adjust thread count proportionally to hardware chonkiness
    numThreads = 14
    pool = Pool(numThreads)

    log = logging.getLogger(timestep.__name__)

    msg = ('Beginning {} experiment with {} samples per inference with {} repeats running on {} threads.'.format(
                timestep.__name__,
                SAMPLES_PER_INFERENCE,
                TIMESTEP_REPEATS,
                numThreads
            ))
    log.info(msg)
    print(msg)
    accuracies = list(
            tqdm(
                pool.imap(timestep, list(range(TIMESTEP_REPEATS))), 
                total=TIMESTEP_REPEATS
                )
            )
    log.critical('### EXPERIMENT COMPLETE ###')
    log.critical('Accuracies: {}'.format(accuracies))
    log.critical('Mean accuracy: {}'.format(np.mean(accuracies)))

    filename = 'log/{}-{}.log'.format(timestep.__name__, START_TIME)
    print('Logged run to {}'.format(filename))

def initActionPreds():
    header = 'gameMap, player, reference_timestep, goal, pred_actions, true_actions'
    for i in range(2):
        log = logging.getLogger('actionpred_timestep{}'.format(i+1))
        log.info(header)
    return

def main():
    initActionPreds()
    runExperiment(timestep1)
    runExperiment(timestep2)
    #runExperiment(timestep3)

if __name__ == '__main__':
    main()

