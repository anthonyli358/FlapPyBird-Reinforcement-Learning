# dqn.py
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from config import config


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)


class DQN:
    def __init__(self, train):
        """
        Learning rate: Higher is faster but more likely to miss optima
        Network size: 4 inputs, 2 outputs - 256 is very precise but longer to train. Higher we probably overfit
        Discount factor: Higher means values survival further into the future
        Target update freq: Higher means more stable targets but slower adaptation
        Batch size: Higher is slower but with smoother gradients
        """

        self.train = train
        self.discount_factor = 0.99
        self.alpha = 1e-4  # adam learning rate
        self.epsilon = 0.01
        self.epsilon_decay = 0.00001
        self.batch_size = 64
        self.target_update_freq = 5000  # steps between target net sync
        self.reward = {0: 0.001, 1: -1}

        self.episode = 0
        self.step_count = 0
        self.scores = []
        self.max_score = 0
        self.moves = []  # current episode transitions

        # Replay buffer
        self.replay_buffer = deque(maxlen=100_000)

        # Networks
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")  # tiny model, use cpu to avoid copying
        self.policy_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.alpha)

        self.previous_state = None
        self.previous_action = 0

        self.load_model()
        self.load_training_states()

    def get_state(self, x, y, vel, pipe):
        pipe0, pipe1 = pipe[0], pipe[1]
        if x - pipe[0]["x"] >= 50:
            pipe0 = pipe[1]
            if len(pipe) > 2:
                pipe1 = pipe[2]

        x0 = pipe0["x"] - x
        y0 = pipe0["y"] - y

        if x0 <= 10:
            y1 = pipe1["y"] - y
        else:
            y1 = 0

        # Raw values, normalised roughly to [-1, 1]
        return [x0 / 140.0, y0 / 180.0, vel / 10.0, y1 / 180.0]

    def act(self, x, y, vel, pipe):
        state = self.get_state(x, y, vel, pipe)

        if self.train and self.previous_state is not None:
            self.replay_buffer.append((self.previous_state, self.previous_action, 0.001, state))
            self.step_count += 1
            if self.step_count % 4 == 0:  # train every 4th frame for speed
                self._train_batch()
            if self.step_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        self.previous_state = state

        if self.train and random.random() < self.epsilon:
            self.previous_action = random.randint(0, 1)
            return self.previous_action

        with torch.no_grad():
            state_t = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q_values = self.policy_net(state_t)
            self.previous_action = q_values.argmax(dim=1).item()

        return self.previous_action

    def update_qvalues(self, score):
        """
        History replay buffer removed

        Args:
            score (_type_): _description_
        """
        self.episode += 1
        self.scores.append(score)
        self.max_score = max(score, self.max_score)

        if not self.train:
            return

        # Penalise last 2 transitions
        for i in range(1, min(3, len(self.replay_buffer) + 1)):
            idx = len(self.replay_buffer) - i
            s, a, r, ns = self.replay_buffer[idx]
            self.replay_buffer[idx] = (s, a, self.reward[1], ns)

        self.previous_state = None  # reset for next episode

        if self.epsilon > 0:
            self.epsilon = max(self.epsilon - self.epsilon_decay, 0)

    def _train_batch(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(
            actions, dtype=torch.long, device=self.device
        ).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(
            next_states, dtype=torch.float32, device=self.device
        )

        q_values = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1).values
            targets = rewards_t + self.discount_factor * next_q

        loss = nn.functional.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)  # gradient clipping
        self.optimizer.step()

    def end_episode(self, score):
        self.episode += 1
        self.scores.append(score)
        self.max_score = max(score, self.max_score)
        self.previous_state = None

    def load_model(self):
        print("Loading DQN model...")
        try:
            checkpoint = torch.load(config["dqn_model_file"], map_location=self.device)
            self.policy_net.load_state_dict(checkpoint["policy_net"])
            self.target_net.load_state_dict(checkpoint["target_net"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        except (IOError, FileNotFoundError):
            print("No existing model found, starting fresh.")

    def save_qvalues(self):
        if self.train:
            print(f"Saving DQN model at episode {self.episode}...")
            torch.save(
                {
                    "policy_net": self.policy_net.state_dict(),
                    "target_net": self.target_net.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                },
                config["dqn_model_file"],
            )

    def load_training_states(self):
        if self.train:
            print("Loading training states...")
            try:
                with open(config["dqn_scores_file"], "r") as f:
                    training_state = json.load(f)
                self.episode = training_state["episodes"][-1]
                self.scores = training_state["scores"]
                self.epsilon = max(self.epsilon - self.epsilon_decay * self.episode, 0)
                self.max_score = max(self.scores)
            except (IOError, FileNotFoundError):
                pass

    def save_training_states(self):
        if self.train:
            print(f"Saving training states with {self.episode} episodes...")
            with open(config["dqn_scores_file"], "w") as f:
                json.dump(
                    {
                        "episodes": [i + 1 for i in range(self.episode)],
                        "scores": self.scores,
                    },
                    f,
                )
