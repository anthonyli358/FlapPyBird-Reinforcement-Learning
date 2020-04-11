import json


class QLearning:
    """
    A Q-Learning agent.

    Load the Q-Learning agent from a file (data/q_values.json by default).
    To train a new agent specify a new file.
    """
    def __init__(self, train):
        """
        Initialise the agent
        :param train: train or run
        """
        self.train = train  # train or run
        self.discount_factor = 0.95  # q-learning discount factor
        self.alpha = 0.7  # learning rate
        self.epsilon = 0.1  # chance to explore vs take local optimum
        self.reward = {0: 0, 1: -1000}  # reward function, focus on only not dying

        # Stabilize and converge to optimal policy
        self.alpha_decay = 0.00001
        self.epsilon_decay = 0.00001  # 1 million episodes to not explore anymore

        # Save states
        self.episode = 0
        self.previous_action = 0
        self.previous_state = "0_0_0_0"  # initial position (x0, y0, vel, y1)
        self.moves = []
        self.scores = []

        # Load states
        self.q_values = {}  # q-table[state][action] decides which action to take by comparing q-values
        self.load_qvalues()
        self.load_training_states()

    def load_training_states(self):
        """Load current training state from json file."""
        try:
            with open("data/training_values.json", "r") as f:
                training_state = json.load(f)
                self.episode = training_state['episodes'][-1]
                self.scores = training_state['scores']
        except IOError:
            pass

    def load_qvalues(self):
        """Load q values and from json file."""
        try:
            with open("data/q_values.json", "r") as f:
                self.q_values = json.load(f)
        except IOError:
            self.init_qvalues(self.previous_state)

    def init_qvalues(self, state):
        """
        Initialise q values if state not yet seen.
        :param state: current state
        :return: None
        """
        if self.q_values.get(state) is None:
            self.q_values[state] = [0, 0, 0]  # [Q of no action, Q of flap action, Number of actions]

    def act(self, x, y, vel, pipe):
        """
        Agent performs an action within the FlapPyBird environment.
        :param x: bird x
        :param y: bird y
        :param vel: bird y velocity
        :param pipe: pipe
        :return: action to take
        """
        # store the transition from previous state to current state
        state = self.get_state(x, y, vel, pipe)
        if self.train:
            self.moves.append((self.previous_state, self.previous_action, state))  # add the experience to history
            self.previous_state = state  # update the last_state with the current state

        # Best action with respect to current state, do nothing is the default
        self.previous_action = 0 if self.q_values[state][0] >= self.q_values[state][1] else 1

        return self.previous_action

    def update_qvalues(self, score):
        """Update q values using history."""
        self.episode += 1
        self.scores.append(score)

        if self.train:
            history = list(reversed(self.moves))
            # Flag if the bird died in the top pipe, don't flap if this is the case
            high_death_flag = True if int(history[0][2].split("_")[1]) > 120 else False
            t, last_flap = 0, True
            for move in history:
                t += 1
                state, action, new_state = move
                self.q_values[state][2] += 1  # make 1 more move
                curr_reward = self.reward[0]
                # Select reward
                if t <= 2:
                    # Penalise last 2 states before dying
                    curr_reward = self.reward[1]
                    if action:
                        last_flap = False
                elif (last_flap or high_death_flag) and action:
                    # Penalise flapping
                    curr_reward = self.reward[1]
                    last_flap = False
                    high_death_flag = False

                self.q_values[state][action] = (1 - self.alpha) * (self.q_values[state][action]) + \
                                               self.alpha * (curr_reward + self.discount_factor *
                                                             max(self.q_values[new_state][0:2]))

            # Decay values for convergence
            # Don't want to decay alpha at a fixed rate
            # if self.alpha > 0:
            #     self.alpha -= self.alpha_decay
            if self.epsilon > 0:
                self.epsilon -= self.epsilon_decay

            self.moves = []  # clear history after updating strategies

    def get_state(self, x, y, vel, pipe):
        """
        Get current state of bird in environment.
        :param x: bird x
        :param y: bird y
        :param vel: bird y velocity
        :param pipe: pipe
        :return: current state (x0_y0_v_y1) where x0 and y0 are diff to pipe0 and y1 is diff to pipe1
        """

        # Get pipe coordinates
        pipe0, pipe1 = pipe[0], pipe[1]
        if x - pipe[0]["x"] >= 50:
            pipe0 = pipe[1]
            if len(pipe) > 2:
                pipe1 = pipe[2]

        x0 = pipe0["x"] - x
        y0 = pipe0["y"] - y
        if -50 < x0 <= 0:
            y1 = pipe1["y"] - y
        else:
            y1 = 0

        # Evaluate player position compared to pipe
        if x0 < -40:
            x0 = int(x0)
        elif x0 < 140:
            x0 = int(x0) - (int(x0) % 10)
        else:
            x0 = int(x0) - (int(x0) % 70)

        if -180 < y0 < 180:
            y0 = int(y0) - (int(y0) % 10)
        else:
            y0 = int(y0) - (int(y0) % 60)

        if -180 < y1 < 180:
            y1 = int(y1) - (int(y1) % 10)
        else:
            y1 = int(y1) - (int(y1) % 60)

        state = str(int(x0)) + "_" + str(int(y0)) + "_" + str(int(vel)) + "_" + str(int(y1))
        self.init_qvalues(state)
        return state

    def save_qvalues(self):
        """Save q values to json file."""
        print(f"Saving Q-table with {len(self.q_values.keys())} to file...")
        with open("data/q_values.json", "w") as f:
            json.dump(self.q_values, f)
        print("Done.")

    def save_training_states(self):
        """Save current training state to json file."""
        print(f"Saving training states with {self.episode} episodes to file...")
        with open("data/training_values.json", "w") as f:
            json.dump({'episodes': [i+1 for i in range(self.episode)],
                       'scores': self.scores}, f)
        print("Done.")
