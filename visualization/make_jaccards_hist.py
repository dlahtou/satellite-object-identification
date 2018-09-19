"""
Histogram generator for model performance visualization
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def make_jaccards_hist():
    '''
    Loads an array containing [intersections, unions] of test masks and model outputs.
    '''

    arrays = np.load('/home/dlahtou/jaccards.npy')
    print(arrays.shape)

    iou = np.divide(arrays[0], arrays[1])

    print(np.mean(iou))

    fig = plt.figure(figsize=[15, 8])
    plt.hist(iou, bins=np.arange(0, 1.1, 0.1), color='#63d297', edgecolor='#4ba173')
    sns.despine()
    plt.title('Histogram of Jaccard Indices for 400 Predictions', fontsize=30)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    fig.savefig('jaccard_hist.png', transparent=True)
    plt.show()
