"""
Exact, side-effect-free game dynamics to simulate future states.

A `pipe` is a `(x_left, gap_center)` tuple.
A `state` for planning is the bird `(y, vel)` plus the committed list of pipes ahead.
"""

import numpy as np
from . import config as C


def bird_step(y: int, vel: int, flap: bool) -> tuple[int, int]:
    """
    Advance the bird one frame. based on flappy.py

    Args:
        y: Bird top-left y in pixels (screen y grows downward).
        vel: Current vertical velocity in pixels/frame.
        flap: Whether the flap action is taken this frame.

    Returns:
        The next ``(y, vel)``.
    """
    if flap:
        vel = C.FLAP_ACC
    else:
        vel = min(vel + C.GRAVITY, C.MAX_VEL_Y)
    return y + vel, vel


def scroll(pipes: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Scroll every pipe left by one frame."""
    return [(px + C.PIPE_VEL_X, gc) for px, gc in pipes]


def clearance(y: int, pipes: list[tuple[int, int]]) -> float:
    """
    Calculates the vertical margin to the nearest lethal boundary.

    Args:
        y: Bird top-left y.
        pipes: Committed pipes ahead of and around the bird.

    Returns:
        The minimum clearance in pixels (negative on collision).
    """
    m = min(y, C.BASE_Y - 1 - (y + C.PLAYER_H))  # ceiling and ground room
    bl, br = C.PLAYER_X, C.PLAYER_X + C.PLAYER_W
    for px, gc in pipes:
        if br > px and bl < px + C.PIPE_W:  # x-overlap
            gap_top = gc - C.PIPE_GAP // 2
            gap_bot = gc + C.PIPE_GAP // 2
            m = min(m, y - gap_top, gap_bot - (y + C.PLAYER_H))
    return float(m)


def alive(y: int, pipes: list[tuple[int, int]]) -> bool:
    """True if the bird is not colliding this frame."""
    return clearance(y, pipes) >= 0


def next_pipes(pipes: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Pipes still ahead of (or overlapping) the bird, nearest first."""
    ahead = [p for p in pipes if p[0] + C.PIPE_W > C.PLAYER_X]
    ahead.sort()
    return ahead


def features(y: int, vel: int, pipes: list[tuple[int, int]]) -> np.ndarray:
    """Continuous network input for state ``(y, vel, pipes)``.

    Uses the two nearest gaps so the policy can anticipate consecutive-gap
    transitions rather than acting greedily on one pipe.

    Returns:
        Float32 array ``[dx1, dy1, vel, dx2, dy2]``, normalised to ~[-1, 1].
    """
    ahead = next_pipes(pipes)
    p1 = ahead[0] if ahead else (C.SCREEN_W, (C.GAP_CENTER_MIN + C.GAP_CENTER_MAX) // 2)
    p2 = ahead[1] if len(ahead) > 1 else p1
    return np.array(
        [
            (p1[0] - C.PLAYER_X) / C.SCREEN_W,
            (p1[1] - y) / C.SCREEN_H,
            vel / C.MAX_VEL_Y,
            (p2[0] - C.PLAYER_X) / C.SCREEN_W,
            (p2[1] - y) / C.SCREEN_H,
        ],
        dtype=np.float32,
    )


def norm_clearance(c: float) -> float:
    """Map a pixel clearance to a bounded value target in [-1, 1]."""
    return float(np.clip(c / C.CLEAR_SCALE, -1.0, 1.0))
