import json
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt


def load_data(filename: str) -> dict:
    """load training results and compute max_score."""
    with open(f"data/{filename}.json", "r") as f:
        training_state = json.load(f)
    max_reached = 0
    training_state['max_scores'] = []
    for i in training_state['scores']:
        max_reached = max(i, max_reached)
        training_state['max_scores'].append(max_reached)
    return training_state


def plot_performance(agent_states: dict, window=50, xlim=None, ylim=None, logy=False) -> None:
    """Plot the training performance."""
    episodes, scores, max_scores = agent_states['episodes'], agent_states['scores'], agent_states['max_scores']
    f, ax = plt.subplots()
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Episode', fontsize=12)
    if logy:
        ax.set_yscale('log')
        plt.ylabel('log(Score)', fontsize=12)
        scores = [x+1 for x in scores]
        max_scores = [x+1 for x in max_scores]
    plt.scatter(episodes, scores, label='scores', color='b', s=3)
    plt.plot(episodes, max_scores, label='max_score', color='g')
    plt.plot(episodes, np.convolve(scores, np.ones((window,)) / window, mode='same'),
             label='rolling_mean_score', color='orange')
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    filename = 'training_values_resume'
    agent_performance = load_data(filename)
    plot_performance(agent_performance, logy=True)
