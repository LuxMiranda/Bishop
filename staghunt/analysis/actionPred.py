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

def getNextPos(pos, action):
    (row,col) = pos
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
def getRelativeAction(agentCurPos, agentNextPos, otherAgentPos):
    origDistance  = getDistance(agentCurPos, otherAgentPos)
    newDistance   = getDistance(agentNextPos, otherAgentPos)
    if newDistance < origDistance:
        return 'T'
    elif newDistance == origDistance:
        return 'S'
    else:
        return 'A'

def getRelativeActions(agent, otherAgent, actionSeq, gameMap):
    relActions    = ''
    agentCurPos   = getPos(agent, gameMap)
    otherAgentPos = getPos(otherAgent, gameMap)
    for action in actionSeq:
        agentNextPos = getNextPos(agentCurPos, action)
        relActions  += getRelativeAction(agentCurPos, agentNextPos, otherAgentPos)
        agentCurPos  = agentNextPos
    return relActions


def getPlayerStates(gameMap):
    states = {}
    config = configparser.ConfigParser()
    config.read('../{}_PlayerA.ini'.format(gameMap))
    states['A'] = int(config.get('MapParameters', 'StartingPoint'))
    config.read('../{}_PlayerB.ini'.format(gameMap))
    states['B'] = int(config.get('MapParameters', 'StartingPoint'))
    config.read('../{}_PlayerC.ini'.format(gameMap))
    states['C'] = int(config.get('MapParameters', 'StartingPoint'))
    #states['Stag1']
    return states

def getStates(file):
    maps = pd.read_csv(file)['gameMap']
    maps = (list(set(list(maps))))
    states = {}
    for gmap in maps:
        states[gmap] = getPlayerStates(gmap)
    return states

def getConvertedData(file):
    states = getStates(file)
    data = pd.read_csv(file)
    data = data.fillna('NNN')
    for i in data.index:
        player   = data.at[i, 'player']
        true_actions  = data.at[i, 'true_actions']
        pred_actions  = true_actions[0] + data.at[i, 'pred_actions']
        stateMap = states[data.at[i, 'gameMap']]
        if player != 'A':
            data.at[i, 'true_relative_to_A'] = getRelativeActions(player, 'A', true_actions, stateMap)
            data.at[i, 'pred_relative_to_A'] = getRelativeActions(player, 'A', pred_actions, stateMap)
        if player != 'B':
            data.at[i, 'true_relative_to_B'] = getRelativeActions(player, 'B', true_actions, stateMap)
            data.at[i, 'pred_relative_to_B'] = getRelativeActions(player, 'B', pred_actions, stateMap)
        if player != 'C':
            data.at[i, 'true_relative_to_C'] = getRelativeActions(player, 'C', true_actions, stateMap)
            data.at[i, 'pred_relative_to_C'] = getRelativeActions(player, 'C', pred_actions, stateMap)
    return data

def accuracyScore(left, right):
    return 1 if left == right else 0

def relativeAccuracy(fromt, tot, log):
    t = tot
    if tot == 2:
        t = -1
    data = getConvertedData(log)
    numCorrect = 0
    total = 0
    for i in data.index:
        if data.at[i, 'player'] != 'A':
            numCorrect += accuracyScore(data.at[i, 'true_relative_to_A'][t], data.at[i, 'pred_relative_to_A'][t])
        if data.at[i, 'player'] != 'B':
            numCorrect += accuracyScore(data.at[i, 'true_relative_to_B'][t], data.at[i, 'pred_relative_to_B'][t])
        if data.at[i, 'player'] != 'C':
            numCorrect += accuracyScore(data.at[i, 'true_relative_to_C'][t], data.at[i, 'pred_relative_to_C'][t])
        total += 2
    print('Total relative accuracy, T{}->T{}: {}'.format(fromt+1, tot+1, numCorrect/total))
    return

def main():
    rawAccuracy1to2()
    rawAccuracy1to3()
    rawAccuracy2to3()
    log = '../log/OVERNIGHT_1_actionpred1-20201028-005016.csv'
    relativeAccuracy(0, 1, log)
    relativeAccuracy(0, 2, log)
    log = '../log/OVERNIGHT_2_actionpred2-20201028-233553.csv'
    relativeAccuracy(1, 2, log)

if __name__ == '__main__':
    main()
