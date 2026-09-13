"""
Real-time, no-peek controller and a live-game adapter which applies the guardian kernel per frame.
"""

import os

import numpy as np

from . import config as C
from . import dynamics as D
from . import viability as V

DEFAULT_KERNEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel.npy")


def _governing(pipes: list[tuple[int, int]]):
    """
    Nearest not-yet-passed on-screen pipe and frames until it clears the bird.

    Returns:
        `(gap_center, px, horizon)` for the governing pipe, or `None` if no
        pipe is ahead (start of the game).
    """
    ahead = [
        (px, gc) for px, gc in pipes if px + C.PIPE_W > C.PLAYER_X and px < C.SCREEN_W
    ]  # unpassed AND on screen
    if not ahead:
        return None
    px, gc = min(ahead)
    horizon = max(1, int(np.ceil((px + C.PIPE_W - C.PLAYER_X) / abs(C.PIPE_VEL_X))))
    return gc, px, horizon


def _safe_sets(gc: int, px: int, horizon: int, R: np.ndarray) -> list[np.ndarray]:
    """
    Per-frame safe sets B of `(y, vel)` states at frame ``t`` from which the bird can
    clear a pipe and be inside the kernel `R` once it clears the pipe. 

    Returns:
        `B[0..horizon]` for the current governing pipe.
    """
    B = [None] * (horizon + 1)
    B[horizon] = R.copy()
    for t in range(horizon - 1, -1, -1):
        pxt = px + C.PIPE_VEL_X * t
        overlap = (pxt < C.PLAYER_X + C.PLAYER_W) and (pxt + C.PIPE_W > C.PLAYER_X)
        ys = np.arange(V.Y_MAX + 1)
        feas = (ys >= 0) & (ys <= V.Y_MAX)
        if overlap:
            feas &= (ys >= gc - V.HALF) & (ys + C.PLAYER_H <= gc + V.HALF)
        succ = np.zeros_like(R)
        for flap in (0, 1):
            y2, v2, ok = V.successors()[flap]
            succ |= B[t + 1][y2, v2] & ok
        B[t] = succ & feas[:, None]
    return B


def robust_action(
    y: int, vel: int, pipes: list[tuple[int, int]], proposal: int, R: np.ndarray
) -> int:
    """
    Return the original `proposal` action if it stays in the kernel-reaching safe set, else a
    safe action. Falls back to whichever action has greater immediate clearance if there's no passable states.

    Args:
        y, vel: Live bird state.
        pipes: On-screen pipes with their revealed gap centres.
        proposal: Action from the policy net or center-seeking (0/1).
        R: Precomputed viability kernel.

    Returns:
        A safe action (0/1).
    """
    gov = _governing(pipes)
    if gov is None:  # open field: stay in the kernel
        best_a, best = proposal, -1
        for a in (proposal, 1 - proposal):
            ny, nvel = D.bird_step(y, vel, bool(a))
            if V.in_kernel(R, ny, nvel):
                return a
            if ny not in (best,):
                best_a = a
        return best_a

    gc, px, horizon = gov
    B = _safe_sets(gc, px, horizon, R)
    best_a, best_c = proposal, -1e9
    for a in (proposal, 1 - proposal):
        ny, nvel = D.bird_step(y, vel, bool(a))
        if (
            0 <= ny <= V.Y_MAX
            and V.V_MIN <= nvel <= V.V_MAX
            and B[1][ny, nvel - V.V_MIN]
        ):
            return a
        # Fallback
        c = D.clearance(ny, D.scroll(pipes))
        if c > best_c:
            best_a, best_c = a, c
    return best_a


class GameAdapter:
    """
    Translate a live FlapPyBird frame into states for the guardian.

    The live engine exposes the bird's `playery` / `playerVelY` and the list
    of on-screen lower pipes. This adapter derives each gap centre for spawned pipes.
    """

    def __init__(self, kernel_path: str | None = None):
        self.R = np.load(kernel_path or DEFAULT_KERNEL)

    def state(self, playery: int, player_vel_y: int, lower_pipes) -> tuple:
        """
        Build `(y, vel, pipes)` from live values.

        Args:
            playery: Bird top-left y from the engine.
            player_vel_y: Bird vertical velocity from the engine.
            lower_pipes: Iterable of dicts/objects with ``x`` and ``y`` (the
                lower pipe's top); the gap centre is ``y - PIPE_GAP/2``.

        Returns:
            The planning state, with pipes filtered to those on screen.
        """
        pipes = []
        for p in lower_pipes:
            px = int(p["x"] if isinstance(p, dict) else p.x)
            lower_top = int(p["y"] if isinstance(p, dict) else p.y)
            if px < C.SCREEN_W:  # on screen only -> gap already revealed
                pipes.append((px, lower_top - C.PIPE_GAP // 2))
        return int(playery), int(player_vel_y), pipes

    def action(self, playery, player_vel_y, lower_pipes, proposal) -> int:
        """
        Map live states and shield a proposal.
        `proposal` is either an action (0/1) or a callable `f(y, vel, pipes) -> 0/1` (from a distlled net).
        """
        y, vel, pipes = self.state(playery, player_vel_y, lower_pipes)
        if callable(proposal):
            proposal = int(proposal(y, vel, pipes))
        return robust_action(y, vel, pipes, proposal, self.R)
