import pandas as pd
import numpy as np

def rawAccuracy():
    t1data = pd.read_csv('../log/OVERNIGHT_1_actionpred1-20201028-005016.csv')
    # Extract pred/true actions for timestep 2
    t1data['pred_t2'] = t1data['pred_actions'].apply(lambda x: str(x)[0])
    t1data['true_t2'] = t1data['true_actions'].apply(lambda x: str(x)[1])
    t1data['correct_pred'] = t1data.apply(lambda x: 1 if x['pred_t2'] == x['true_t2'] else 0, axis=1)
    totalCorrect = np.sum(t1data['correct_pred'])
    totalPreds = t1data.shape[0]
    accuracy = totalCorrect/totalPreds
    print('Timestep 1 raw accuracy: {:.2%}'.format(accuracy) +
        (' (Baseline: {:.2%})'.format(1.0/5.0)))

    return


def main():
    rawAccuracy()

if __name__ == '__main__':
    main()
