# FlapPyBird-RL

Exploration implementing reinforcement learning using [Q-learning](https://en.wikipedia.org/wiki/Q-learning) in [Flappy Bird](https://en.wikipedia.org/wiki/Flappy_Bird).

## Results

### Initial Training

The agent was initially trained for around 100,000 episodes without any exploration and the learning rate alpha constant. The reward function is to penalise -1000 for a death and 0 otherwise, such that the agent is able to learn to perform better than the finite value setting a positive reward would give (e.g. rewarding +1 for a score increase means the agent would maximise the chance of getting more than 1000). Through undertaking this project the most difficult part was defining a good reward function and how this links to the agent learning alongside other techniques.



### Experience Replay: Catastrophic Forgetting

Although the initial training performed well, it was very slow to improve further - it takes a very long time for it to reach a scenario it fails at. By introducting experience replay the agent can attempt the difficult scenario multiples times to attempt to overcome it. The lenght of replay was set to 70 (the distance between pipes), and it tries until it passes the difficult scenario or appears to be stuck in a resume loop (100 attempts). Upon passing the difficult scenario, upon failure it restarts from the beginning to avoid the maximum score reached from continously increasing.

When trying to overcome the rarer scenarios the agent has 'forgotten' what was originally learnt, leading to a drop in agent performance as it fails to return to the previously generalisable action states. This is known as catastrophic forgetting which typically leads to an oscillation in agent performance as it unlearns and relearns the optimal actiosn to take. This is compounded by repeatedly learning from the same scenario failure has also contributed to this, leading to overfitting of the problem.

### Experience Replay: Replay Buffer

To overcome catastrophic forgetting, alpha is decayed as the agent is trained, helping it to retain the information it learn early on whilst still learning from rarer scenarios. In addition, the number of attempts to be considered stuck in a loop is reduced to 50 and during experience replay we create a 'replay buffer' with all the actions taken in the attempts to overcome a scenario. The Q-table is updated from this in a mini-batch fashion, sampling 5 of attempts remaining once the agent has overcome the scenario or is considered stuck in a reply loop. To avoid overfitting the agent doesn't replay the scenario until success.

### Epsilon Greedy Policy

We now try freshly trained agent introducing the exploration rate epsilon that gives a chance to explore until it decays to 0 after 10,000 episodes, and alpha decay which decays alpha to 0.1 after 20,000 episodes from the beginning of training.  

### Validation

### Future Work

- Longer training times, the best performing agent was trained for a total of 15 hours and only reached ... episodes
- Implement prioritized experience replay
- Train an agent which never dies in the Flappy Bird environment

## Getting Started

Added modules:
- [anaysis.py](analysis.py) Analysis file for investigating agent performance
- [config.py](config.py) Config file for changing the agent training parameters
- [flappy_rl.py](flappy_rl.py): [FlapPyBird](https://github.com/sourabhv/FlapPyBird) implementation with agent training/runner code included
- [q_learning.py](q_learning.py): An implementation of a Q-learning agent class made with reference to [rl-flappybird](https://github.com/kyokin78/rl-flappybird)

Change the training parameters in [config.py](config.py) and run the [flappy_rl.py](flappy_rl.py) module.

## Development

[q_learning.py](q_learning.py)
- Q-learning is performed based upon the states [x0, y0, vel, y1], where x0 and y0 are the player distances to the next lower pipe, 
vel is the agent y velocity, and y1 is the y distance between the lower pipes. x0, y0, y1 are calculated from the playerx, playery, and the array of lower pipes
- States are added to the Q-table as they are encountered rather than initialising a sparse Q-table.
The initial state is initialised to [0, 0, 0] where the array represents [Q of no action, Q of flap action, Times experienced this state]
- Alpha (learning date) decay is added to prevent overfitting and reduce the chance of catastrophic forgetting as training continues
- An epsilon greedy policy to give a chance to explore has been added but commented out. It was found that 
exploration is not efficient or required for this agent (only 2 possible states, flap or no flap) and environment (repeating)
- Improved performance by adding functions to reduce the number of moves in memory for updating the Q-table, 
and to update the Q-table and end the episode if the maximum score is reached

[flappy_rl.py](flappy_rl.py)
- Remove sounds, welcome animation, and game over screen to improve performance
- Added the ability to perform runs without game rendering, greatly improving runtime
- Added the ability to resume the game from 70 frames (distance between pipes) before death. 
This enables the agent to learn to overcome scenarios not often encountered. 
Once the agent has overcome this scenario, upon its next death it restarts training from the beginning 
to avoid the maximum score reached from continuously increasing

## Forked From [FlapPyBird](https://github.com/sourabhv/FlapPyBird)

A Flappy Bird Clone made using [python-pygame][pygame]

### How-to (as tested on MacOS)

1. Install Python 3.x (recommended) 2.x from [here](https://www.python.org/download/releases/)

2. Install [pipenv]

2. Install PyGame 1.9.x from [here](http://www.pygame.org/download.shtml)

3. Clone the repository:

```bash
$ git clone https://github.com/sourabhv/FlapPyBird
```

or download as zip and extract.

4. In the root directory run

```bash
$ pipenv install
$ pipenv run python flappy.py
```

5. Use <kbd>&uarr;</kbd> or <kbd>Space</kbd> key to play and <kbd>Esc</kbd> to close the game.

(For x64 windows, get exe [here](http://www.lfd.uci.edu/~gohlke/pythonlibs/#pygame))

### Notable forks

- [FlappyBird Fury Mode](https://github.com/Cc618/FlapPyBird)
- [FlappyBird Model Predictive Control](https://github.com/philzook58/FlapPyBird-MPC)
- [FlappyBird OpenFrameworks Port](https://github.com/TheLogicMaster/ofFlappyBird)

Made something awesome from FlapPyBird? Add it to the list :)

### ScreenShot

![Flappy Bird](screenshot1.png)

[pygame]: http://www.pygame.org
[pipenv]: https://pipenv.readthedocs.io/en/latest/
