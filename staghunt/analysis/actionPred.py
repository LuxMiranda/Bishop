import pandas as pd
import numpy as np
import configparser

def rawAccuracy1to2():
    t1data = pd.read_csv('../log/OVERNIGHT_1_actionpred1-20201028-005016.csv')
    # Extract pred/true actions for timestep 2
    t1data['pred_t2'] = t1data['pred_actions'].apply(lambda x: str(x)[0])
    t1data['true_t2'] = t1data['true_actions'].apply(lambda x: str(x)[1])
    t1data['correct_pred'] = t1data.apply(lambda x: 1 if x['pred_t2'] == x['true_t2'] else 0, axis=1)
    totalCorrect = np.sum(t1data['correct_pred'])
    totalPreds = t1data.shape[0]
    accuracy = totalCorrect/totalPreds
    print('Timestep 1 raw accuracy for timestep 2: {:.2%}'.format(accuracy) +
        (' (Baseline: {:.2%})'.format(1.0/5.0)))
    return

def rawAccuracy1to3():
    t1data = pd.read_csv('../log/OVERNIGHT_1_actionpred1-20201028-005016.csv')
    # Extract pred/true actions for timestep 2
    t1data['pred_t2'] = t1data['pred_actions'].apply(lambda x: str(x)[1])
    t1data['true_t2'] = t1data['true_actions'].apply(lambda x: str(x)[-1])
    t1data['correct_pred'] = t1data.apply(lambda x: 1 if x['pred_t2'] == x['true_t2'] else 0, axis=1)
    totalCorrect = np.sum(t1data['correct_pred'])
    totalPreds = t1data.shape[0]
    accuracy = totalCorrect/totalPreds
    print('Timestep 1 raw accuracy for timestep 3: {:.2%}'.format(accuracy) +
        (' (Baseline: {:.2%})'.format(1.0/5.0)))
    return


def rawAccuracy2to3():
    t2data = pd.read_csv('../log/OVERNIGHT_2_actionpred2-20201028-233553.csv')
    # Extract pred/true actions for timestep 2
    t2data['pred_t3'] = t2data['pred_actions'].apply(lambda x: str(x)[0])
    t2data['true_t3'] = t2data['true_actions'].apply(lambda x: str(x)[-1])
    t2data['correct_pred'] = t2data.apply(lambda x: 1 if x['pred_t3'] == x['true_t3'] else 0, axis=1)
    totalCorrect = np.sum(t2data['correct_pred'])
    totalPreds = t2data.shape[0]
    accuracy = totalCorrect/totalPreds
    print('Timestep 2 raw accuracy for timestep 3: {:.2%}'.format(accuracy) +
        (' (Baseline: {:.2%})'.format(1.0/5.0)))
    return


def cellNumToCoords(num):
    return (int(np.floor(num / 7)), (num % 7))

# gameMap has location of each agent stored as a cell number
# gameMap['A'] = 34
# gameMap['B'] = 12
# etc.
def getPos(agent, gameMap):
    return cellNumToCoords(gameMap[agent])

def getNextPos(agent, action, gameMap):
    (row,col) = getPos(agent, gameMap)
    if action == 'L':
        return (row, col-1)
    elif action == 'R':
        return (row, col+1)
    elif action == 'U':
        return (row-1, col)
    elif action == 'D':
        return (row+1, col)
    else:
        return (row, col)

def getDistance(pos1, pos2):
    return np.abs(pos2[0] - pos1[0]) + np.abs(pos2[1] - pos1[1])

# Three states:
#   T - Move   [T]owards
#   A - Move   [A]way
#   S - Remain [S]tationary
def getRelativeAction(agent, otherAgent, action, gameMap):
    agentCurPos   = getPos(agent, gameMap)
    agentNextPos  = getNextPos(agent, action, gameMap)
    otherAgentPos = getPos(otherAgent, gameMap)
    origDistance  = getDistance(agentCurPos, otherAgentPos)
    newDistance   = getDistance(agentNextPos, otherAgentPos)
    if newDistance < origDistance:
        return 'T'
    elif newDistance == origDistance:
        return 'S'
    else:
        return 'A'

def getRelativeActions(agent, otherAgent, actionSeq, gameMap):
    return ''.join([getRelativeAction(agent, otherAgent, actionSeq, gameMap) for a in actionSeq])


def getStates(gameMap):
    states = {}
    config = configparser.ConfigParser()
    config.read('../{}_PlayerA.ini'.format(gameMap))
    states['A'] = int(config.get('MapParameters', 'StartingPoint'))
    config.read('../{}_PlayerB.ini'.format(gameMap))
    states['B'] = int(config.get('MapParameters', 'StartingPoint'))
    config.read('../{}_PlayerC.ini'.format(gameMap))
    states['C'] = int(config.get('MapParameters', 'StartingPoint'))
    return states

def test():
    gameMap = 'StagHunt_a'
    states  = getStates(gameMap)
    print(getRelativeAction('A','B','R', states))
    return

def triStateAccuracy():
    # Timestep 1 to timestep 2
    # Get data
    # Create new columns:
        # true_relative_A
        # true_relative_B
        # true_relative_C
        # true_relative_S1
        # true_relative_S2
    # Compare accuracy
    return

def main():
    rawAccuracy1to2()
    rawAccuracy1to3()
    rawAccuracy2to3()
    triStateAccuracy()

if __name__ == '__main__':
    #main()
    test()
