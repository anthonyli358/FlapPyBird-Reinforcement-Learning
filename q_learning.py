import numpy as np
import json


class QLearning:
    """A Q-Learning Agent."""

    def __init__(self, train):
        """
        Initialise the agent
        :param train: train or run
        """
        self.train = train  # train or run
        self.discount_factor = 0.95  # q-learning discount factor
        self.alpha = 0.6  # learning rate
        self.epsilon = 0.1  # chance to explore vs take local optimum
        self.reward = {0: 0, 1: -1000}  # reward function, focus on only not dying

        # Stabilize and converge to optimal policy
        self.alpha_decay = 0.00001
        self.epsilon_decay = 0.00001

        # States and actions
        self.episode = 0  # game count of current run, incremented after every death
        self.save_state = 25  # save state every 25 iterations
        self.previous_state = (96, 47, 0, 47)  # 4 states to take action on
        self.previous_action = 0
        self.moves = []

        # Initialisation for q value matrix
        self.xdif = 130
        self.ydif = 130
        self.vely = 20
        self.y1 = 130

        # initialize matrix to store qvalues
        self.q_values = np.zeros((self.xdif, self.ydif, self.vely, self.y1, 2))
        self.load_qvalues()

    def load_qvalues(self):
        """Load q values from json file."""
        try:
            f = open("data/qvalues.json", "r")
        except IOError:
            return
        self.q_values = json.load(f)
        f.close()

    def act(self, xdif, ydif, vely, y1):
        """
        Agent performs an action within the FlapPyBird environment.
        :param xdif: x_dist to pipe
        :param ydif: y_dist to pipe
        :param vely: bird y velocity
        :param y1: vertical distance between next two pipes
        :return: action to take
        """
        # store the transition from previous state to current state
        if self.train:
            state = (xdif, ydif, vely, y1)
            self.moves.append((self.previous_state, self.previous_action, state))  # add the experience to history
            self.previous_state = state  # update the last_state with the current state

        # Best action with respect to current state
        if self.q_values[xdif, ydif, vely, y1][0] >= self.q_values[xdif, ydif, vely, y1][1]:
            self.previous_action = 0
        else:
            self.previous_action = 1
        return self.previous_action

    def update_qvalues(self):
        """Update q values using history."""
        self.episode += 1

        if self.train:
            history = list(reversed(self.moves))
            t, last_flap = 0, True
            for move in history:
                state, action, new_state = move
                curr_reward = self.reward[0]
                # Select reward
                if t <= 2:
                    # Penalise last 2 states before dying
                    curr_reward = self.reward[1]
                    if action:
                        last_flap = False
                elif last_flap and action:
                    # Penalise flapping
                    curr_reward = self.reward[1]
                    last_flap = False

                self.q_values[state][action] = (1 - self.alpha) * (self.q_values[state][action]) + \
                                                     self.alpha * (curr_reward + self.discount_factor *
                                                                   max(self.q_values[new_state]))
            self.moves = []

            # Decay values for convergence
            if self.alpha > 0:
                self.alpha -= self.alpha_decay
            if self.epsilon > 0:
                self.epsilon -= self.epsilon_decay

            self.moves = []  # clear history after updating strategies

    def save_qvalues(self, force_save=False):
        """Save q values to json file."""
        if self.episode % self.save_state == 0 or force_save:
            with open("data/qvalues.json", "w") as f:
                json.dump(self.q_values, f)
            print("Q-values updated on local file.")

# TODO: NumPy not json serializable
