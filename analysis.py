import json
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filename: str) -> dict:
    with open(f"data/{filename}.json", "r") as f:
        training_state = json.load(f)
    training_state['max_scores'] = [max(training_state['scores'][:i]) for i in range(len(training_state['scores']))]
    return training_state


def plot_performance(agent_states: dict, xlim=None, ylim=None):
    plt.figure()
    plt.plot(agent_states['episodes'], agent_states['scores'])
    plt.plot(agent_states['episodes'], agent_states['max_score'])
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.show()


if __name__ == '__main__':
    filename = 'training_states'
    agent_performance = load_data('training_states')
    plot_performance(agent_performance)
