import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_current_date_time():
    return datetime.now().strftime('%Y/%m/%d %H:%M:%S')


def plot_scores(scores):
    # plot the scores
    fig = plt.figure()
    _ = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
