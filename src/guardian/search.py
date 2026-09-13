"""
MCTS (Monte Carlo Tree Search) but this environment is deterministic so we
just enumerate the possiblities over a few frames and choose the best action.
"""

from __future__ import annotations

from typing import Callable, Optional

from . import config as C
from . import dynamics as D

LeafValue = Callable[[int, int, list], float]


def _leaf_default(y: int, vel: int, pipes: list) -> float:
    """Fallback leaf evaluation is the normalised current clearance."""
    return D.norm_clearance(D.clearance(y, pipes))


def search(
    y: int,
    vel: int,
    pipes: list[tuple[int, int]],
    depth: int,
    leaf_value: Optional[LeafValue] = None,
) -> tuple[float, int]:
    """
    Run the robust search from a state.
    Recursion ends when the bird crashes or depth = 0.

    Args:
        y, vel, pipes: Exact planning state.
        depth: Frames of lookahead.
        leaf_value: Optional learned evaluator for final search (depth=0)
            leaves; defaults to normalised clearance.

    Returns:
        `(value, action)` safety score of the best plan from this state and the best
        root action (0/1). Action is 0 is the bird is already colliding.
    """
    leaf = leaf_value or _leaf_default
    memo: dict[tuple[int, int, int], float] = {}

    def rec(cy: int, cvel: int, cpipes: list, d: int) -> float:
        if not D.alive(cy, cpipes):
            return C.DEAD_VALUE
        if d == 0:
            return leaf(cy, cvel, cpipes)
        key = (cy, cvel, d)
        cached = memo.get(key)
        if cached is not None:
            return cached
        best = C.DEAD_VALUE
        npipes = D.scroll(cpipes)
        for a in (False, True):
            ny, nvel = D.bird_step(cy, cvel, a)
            if not D.alive(ny, npipes):
                node = C.DEAD_VALUE
            else:
                node = min(
                    D.norm_clearance(D.clearance(ny, npipes)),
                    rec(ny, nvel, npipes, d - 1),
                )
            if node > best:
                best = node
        memo[key] = best
        return best

    if not D.alive(y, pipes):
        return C.DEAD_VALUE, 0
    best_v, best_a = C.DEAD_VALUE, 0
    npipes = D.scroll(pipes)
    for a in (0, 1):
        ny, nvel = D.bird_step(y, vel, bool(a))
        if not D.alive(ny, npipes):
            node = C.DEAD_VALUE
        else:
            node = min(
                D.norm_clearance(D.clearance(ny, npipes)),  # node clearance
                rec(ny, nvel, npipes, depth - 1),
            )
        if node > best_v:
            best_v, best_a = node, a
    return best_v, best_a
