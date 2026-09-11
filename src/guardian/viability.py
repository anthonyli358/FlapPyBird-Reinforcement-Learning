"""
Robust viability kernel - the foundation for an immortal bird.

The online guardian must not read pipes that don't yet exist in the game window.
Instead, we can compute all the safe states offline.
"""

from functools import lru_cache

import numpy as np

from . import config as C

T = C.PIPE_SPACING // abs(C.PIPE_VEL_X)  # nominal frames per cycle (reference only)
Y_MAX = C.BASE_Y - 1 - C.PLAYER_H  # max alive top-left y (inclusive)
V_MIN, V_MAX = C.FLAP_ACC, C.MAX_VEL_Y  # velocity range on the lattice
NV = V_MAX - V_MIN + 1
HALF = C.PIPE_GAP // 2


def _cycle(p0: int) -> tuple[int, list[bool]]:
    """
    Horizon and per-frame overlap schedule for a pipe entering at px `p0`.

    - The horizon is the first frame the bird has cleared the pipe.
    - Each cycle is the bird passing a pipe.

    Returns:
        `(horizon, overlap)` where overlap[t] says whether the pipe
        constrains the bird's y at frame `t` (`0 <= t <= horizon`).
    """
    horizon = max(1, int(np.ceil((p0 + C.PIPE_W - C.PLAYER_X) / abs(C.PIPE_VEL_X))))
    overlap = []
    for t in range(horizon + 1):
        pxt = p0 + C.PIPE_VEL_X * t
        overlap.append(
            (pxt < C.PLAYER_X + C.PLAYER_W) and (pxt + C.PIPE_W > C.PLAYER_X)
        )
    return horizon, overlap


@lru_cache(maxsize=None)
def successors():
    """
    Successor (next state) (y, vel) index tables for both actions (0 = no-flap, 1 = flap).

    Built once from the fixed lattice constants and cached.

    Returns:
        `{action: (y2_index, v2_index, bool_mask)}` where the mask is False
        wherever the successor leaves the alive y-range.
    """
    ys = np.arange(Y_MAX + 1)[:, None]  # (Ny, 1)
    vs = np.arange(V_MIN, V_MAX + 1)[None, :]  # (1, Nv)
    out = {}
    for flap in (0, 1):
        if flap:
            v2 = np.full_like(vs, C.FLAP_ACC) + 0 * ys
        else:
            v2 = np.minimum(vs + C.GRAVITY, C.MAX_VEL_Y) + 0 * ys
        y2 = ys + v2
        ok = (y2 >= 0) & (y2 <= Y_MAX)
        out[flap] = (np.clip(y2, 0, Y_MAX), v2 - V_MIN, ok)
    return out


def _y_feasible(overlap_t: bool, g: int) -> np.ndarray:
    """Checks if the bird is alive at a frame with the given overlap flag and pipe gap `g`."""
    ys = np.arange(Y_MAX + 1)
    ok = (ys >= 0) & (ys <= Y_MAX)
    if overlap_t:
        ok &= (ys >= g - HALF) & (ys + C.PLAYER_H <= g + HALF)
    return ok


def _pre(R: np.ndarray, g: int, p0: int) -> np.ndarray:
    """
    Computes that states given a bird and pipe position from which
    R (passing a pipe) is reachable.

    Args:
        R: Boolean grid (Ny, Nv), the target set at cycle end.
        g: Gap centre for this cycle.
        p0: The pipe's px at cycle start (when it becomes governing).

    Returns:
        Boolean grid of states at cycle start from which R is reachable.
    """
    horizon, overlap = _cycle(p0)
    B = R.copy()  # B at frame `horizon`
    for t in range(horizon - 1, -1, -1):
        succ = np.zeros_like(B)
        for flap in (0, 1):
            y2, v2, ok = successors()[flap]
            succ |= B[y2, v2] & ok  # can this action reach B_{t+1}?
        feas = _y_feasible(overlap[t], g)[:, None]  # (Ny, 1) broadcast over v
        B = succ & feas
    return B


def compute_kernel(
    gaps=None, entries=None, max_iter: int = 300, verbose: bool = False
) -> np.ndarray:
    """
    Compute the kernel `R` as a backward-reachability fixed point.

    Start from the optimistic guess that every state is safe, then shrink to a fixed
    kernal which the set where every surviving state can cleara pipe and return to another surviving state:
      1. Keep a state only if, for *every* possible next pipe, some sequence of
         actions threads that pipe without colliding AND lands back in the current
         kernel.
      2. Applying that test drops the now-unsafe states, leaving a smaller kernel.
      3. Repeat: states whose only escape was into a state we just dropped now fail
         the test too, so the kernel keeps shrinking.
         
    Args:
        gaps: Gap centres to cover; default is every integer in the gap band.
        entries: Entry-x values to cover; default is every integer in the entry band.
        max_iter: Safety cap on iterations.
        verbose: Print the surviving-state count each iteration.

    Returns:
        Boolean grid (Ny, Nv) where non-empty means immortality is achieveable.
    """
    # The centre of the next gap
    if gaps is None:
        gaps = range(C.GAP_CENTER_MIN, C.GAP_CENTER_MAX + 1)  # 
    # x at which next pipe become governing
    if entries is None:
        entries = range(C.ENTRY_PX_MIN, C.ENTRY_PX_MAX + 1)  
    gaps = list(gaps)
    entries = list(entries)
    R = np.ones((Y_MAX + 1, NV), dtype=bool)
    for it in range(max_iter):
        nxt = np.ones_like(R)
        stop = False
        for g in gaps:
            for p0 in entries:
                nxt &= _pre(R, g, p0)
                if not nxt.any():
                    stop = True
                    break
            if stop:
                break
        if verbose:
            print(f"  iter {it}: {int(nxt.sum())} states")
        if np.array_equal(nxt, R):
            return R
        R = nxt
        if not R.any():
            return R
    return R


def in_kernel(R: np.ndarray, y: int, vel: int) -> bool:
    """True if (y, vel) is inside the kernel R (a safe state) and False otherwise."""
    if not (0 <= y <= Y_MAX and V_MIN <= vel <= V_MAX):
        return False
    return bool(R[y, vel - V_MIN])


def build_kernel(save: bool = True, verbose: bool = True) -> np.ndarray:
    """
    Compute the robust viability kernel and (optionally) save it to kernel.npy.

    Returns:
        The boolean grid `R`. This is the go/no-go step for immortality as
        a non-empty `R` means safe states are always reachable.
    """
    import os

    R = compute_kernel(verbose=verbose)
    n = int(R.sum())
    print(
        f"kernel |R| = {n} of {(Y_MAX + 1) * NV} states "
        f"({'NON-EMPTY: immortality achievable without peeking' if n else 'EMPTY'})"
    )
    if n:
        ys, vis = np.where(R)
        print(
            f"  y in [{ys.min()}, {ys.max()}], vel in "
            f"[{vis.min() + V_MIN}, {vis.max() + V_MIN}]"
        )
        if save:
            out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel.npy")
            np.save(out, R)
            print(f"  saved {out}")
    return R


if __name__ == "__main__":
    build_kernel()
