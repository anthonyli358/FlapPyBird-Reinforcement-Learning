# FlapPyBird-RL

Exploration implementing reinforcement learning using [Q-learning](https://en.wikipedia.org/wiki/Q-learning) in [Flappy Bird](https://en.wikipedia.org/wiki/Flappy_Bird).

## Results

## Getting Started

Added modules:
- [flappy_rl.py](flappy_rl.py): [FlapPyBird](https://github.com/sourabhv/FlapPyBird) implementation with agent training/runner code included
- [q_learning.py](q_learning.py): An implementation of a Q-learning agent class made with reference to [rl-flappybird](https://github.com/kyokin78/rl-flappybird)

Select whether to train or run the trained agent by changing the `train` variable at the top of [flappy_rl.py](flappy_rl.py) to `True` or `False`, then run the module.

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
