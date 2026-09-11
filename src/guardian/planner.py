"""
Model-based expert and safety shield used by guardian/expert_iteration.py

Generates proposals for traning how an agent should behave.
"""

from . import config as C
from . import dynamics as D


def center_seeking_action(y: int, vel: int, pipes: list[tuple[int, int]]) -> int:
    """
    Aim if coasting `LOOKAHEAD` frames would bring the bird below
    the lower pipe + offset (aim low on a pipe since aiming for the middle
    would crash into the upper pipe.)

    Args:
        y: Bird top-left y.
        vel: Bird vertical velocity.
        pipes: Committed pipes ahead.

    Returns:
        1 to flap, 0 to coast.
    """
    ahead = D.next_pipes(pipes)
    gc = ahead[0][1] if ahead else (C.GAP_CENTER_MIN + C.GAP_CENTER_MAX) // 2
    yy, vv = y, vel
    for _ in range(C.LOOKAHEAD):
        yy, vv = D.bird_step(yy, vv, False)
    return 1 if (yy + C.PLAYER_H / 2) > gc + C.CENTER_OFFSET else 0


def _witness_survives(
    y: int, vel: int, pipes: list[tuple[int, int]], horizon: int
) -> bool:
    """If centre seeking is succesful then we can slip the full BFS on easy frames."""
    pt = pipes
    for _ in range(horizon):
        a = center_seeking_action(y, vel, pt)
        y, vel = D.bird_step(y, vel, bool(a))
        pt = D.scroll(pt)
        if not D.alive(y, pt):
            return False
    return True


def survivable(y: int, vel: int, pipes: list[tuple[int, int]], horizon: int) -> bool:
    """
    Checks whether some action sequence keeps the bird alive for `horizon` frames.

    Tries the cheap witness first, then falls back to BFS over `(y, vel)` until
    there are no survivable states left or the horizon is cleared.
    """
    if not D.alive(y, pipes):
        return False
    if _witness_survives(y, vel, pipes, horizon):
        return True
    frontier = {(y, vel)}
    pt = pipes
    for _ in range(horizon):
        pt = D.scroll(pt)
        nxt = set()
        for cy, cvel in frontier:
            for a in (False, True):
                ny, nvel = D.bird_step(cy, cvel, a)
                if D.alive(ny, pt):
                    nxt.add((ny, nvel))
        if not nxt:
            return False
        frontier = nxt
    return True


def shield(
    y: int,
    vel: int,
    pipes: list[tuple[int, int]],
    proposed: int,
    horizon: int = C.SHIELD_HORIZON,
) -> int:
    """
    Return the `propsed` action if it survives, else the alternative.

    Args:
        proposed: Action suggested by the expert or policy net (0/1).
        horizon: Frames the resulting state must remain survivable for.

    Returns:
        A safe action (0/1).
    """
    best_a, best_c = proposed, -1e9
    for a in (proposed, 1 - proposed):
        ny, nvel = D.bird_step(y, vel, bool(a))
        npipes = D.scroll(pipes)
        c = D.clearance(ny, npipes)
        if c >= 0 and survivable(ny, nvel, npipes, horizon - 1):
            return a
        if c > best_c:  # track least-bad fallback if truly doomed
            best_a, best_c = a, c
    return best_a


def assert_passable() -> None:
    """
    Sanity-check that consecutive gaps are individually reachable.
    Otherwise immortality is not achieveable in the environment.
    """
    frames = C.PIPE_SPACING // abs(C.PIPE_VEL_X)
    max_drop = C.MAX_VEL_Y * frames
    max_climb = -C.FLAP_ACC * frames  # flapping every frame
    span = C.GAP_CENTER_MAX - C.GAP_CENTER_MIN
    assert max_drop >= span and max_climb >= span, (
        f"gap span {span} unreachable in {frames} frames "
        f"(drop {max_drop}, climb {max_climb}); no controller can be immortal"
    )
