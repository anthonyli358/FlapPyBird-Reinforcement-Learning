"""
Headless, integer-exact Flappy Bird environment without rendering.
"""
import numpy as np

from . import config as C
from . import dynamics as D


class FlappyEnv:
    """
    Minimal headless environment exposing the exact planning state.

    Attributes:
        y: Bird top-left y.
        vel: Bird vertical velocity.
        pipes: Committed pipes ``(x_left, gap_center)``, nearest first.
        score: Pipes passed so far.
    """

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, y0: int | None = None, vel0: int = 0) -> np.ndarray:
        """Start a new episode and return the initial feature vector."""
        self.y = int((C.GAP_CENTER_MIN + C.GAP_CENTER_MAX) // 2 - C.PLAYER_H // 2) if y0 is None else int(y0)
        self.vel = int(vel0)
        self.score = 0
        self.pipes = []
        px = C.SCREEN_W
        for _ in range(C.PIPES_AHEAD):
            self.pipes.append((px, self._gap()))
            px += C.PIPE_SPACING
        return D.features(self.y, self.vel, self.pipes)

    def _gap(self) -> int:
        """Sample an integer gap centre from the configured band."""
        return int(self.rng.integers(C.GAP_CENTER_MIN, C.GAP_CENTER_MAX + 1))

    def _commit_pipes(self) -> None:
        """Drop passed pipes and top up so `PIPES_AHEAD` remain committed."""
        self.pipes = [p for p in self.pipes if p[0] + C.PIPE_W > 0]
        while len(D.next_pipes(self.pipes)) < C.PIPES_AHEAD:
            last_x = max(px for px, _ in self.pipes) if self.pipes else C.SCREEN_W
            self.pipes.append((last_x + C.PIPE_SPACING, self._gap()))

    def step(self, flap: bool) -> tuple[np.ndarray, float, bool]:
        """
        Apply one action.

        Returns:
            ``(features, reward, done)`` where reward is +1 per pipe passed and
            done is True on collision.
        """
        self.y, self.vel = D.bird_step(self.y, self.vel, flap)
        self.pipes = D.scroll(self.pipes)
        done = not D.alive(self.y, self.pipes)
        reward = 0.0
        for px, _ in self.pipes:  # a pipe centre crossing PLAYER_X this frame == a pass
            center = px + C.PIPE_W / 2
            if (center - C.PIPE_VEL_X) >= C.PLAYER_X > center:
                self.score += 1
                reward = 1.0
        self._commit_pipes()
        return D.features(self.y, self.vel, self.pipes), reward, done

    def state(self) -> tuple[int, int, list[tuple[int, int]]]:
        """Return the exact planning state `(y, vel, pipes)`."""
        return self.y, self.vel, list(self.pipes)
