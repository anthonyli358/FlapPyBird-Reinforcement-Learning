"""
On-policy DAgger (Dataset aggreation distillation of the immortal (center-seeking + shield) expert.
"""

import time
import numpy as np

from . import config as C
from . import dynamics as D
from . import planner as P
from .env import FlappyEnv
from .search import search
from . import numpy_verify as base  # net init/forward/act/grads/Adam/train/survival

rng = np.random.default_rng(0)


def expert_action(y, vel, pipes):
    """The smooth immortal expert: center-seeking proposal, shield-corrected."""
    return P.shield(y, vel, pipes, P.center_seeking_action(y, vel, pipes))


def collect(p, frames, seed, on_policy):
    """
    Collect (features, expert_action, value, margin).

    Args:
        on_policy: If True, the net drives (shield rescues only when unsafe).
            If False (warm start), the expert drives.
    """
    X, A, V, M, G = [], [], [], [], []
    env = FlappyEnv(seed=seed)
    env.reset()
    for _ in range(frames):
        y, vel, pipes = env.state()
        lab = expert_action(y, vel, pipes)  # DAgger label everywhere
        net_a = base.act(p, y, vel, pipes)
        if on_policy:
            play = P.shield(y, vel, pipes, net_a)  # net drives, rescued
        else:
            play = lab
        val, _ = search(y, vel, pipes, 6)  # robust value target (no net leaf)
        X.append(D.features(y, vel, pipes))
        A.append(lab)
        V.append(val)
        M.append(D.clearance(y, pipes))
        G.append(int(net_a != lab))  # disagreement flag
        _, _, done = env.step(bool(play))
        if done:
            env.reset()
    return (np.array(X), np.array(A), np.array(V), np.array(M), np.array(G))


def train_dagger(p, opt, data, epochs, batch=256):
    """Train, drawing half of each batch from net!=expert disagreement states."""
    X, A, V, M, G = data
    n = len(X)
    dis = np.where(G == 1)[0]
    for _ in range(epochs):
        for _ in range(max(1, n // batch)):
            npri = batch // 2
            if len(dis):
                pri = np.random.choice(dis, npri)
            else:
                pri = np.random.randint(0, n, npri)
            uni = np.random.randint(0, n, batch - npri)
            idx = np.concatenate([pri, uni])
            c = base.forward(p, X[idx])
            ce, mse, g = base.grads(p, c, A[idx], V[idx])
            opt.step(p, g)
    return ce, mse


def override_rate(p, seeds=range(2), cap=1500):
    """Fraction of frames the shield must change the net's action (deployed cost)."""
    changed = total = 0
    for s in seeds:
        env = FlappyEnv(seed=s)
        env.reset()
        for _ in range(cap):
            y, vel, pipes = env.state()
            na = base.act(p, y, vel, pipes)
            sa = P.shield(y, vel, pipes, na)
            changed += na != sa
            total += 1
            env.step(bool(sa))
    return changed / total


def main():
    """On-policy DAgger distillation; prints bare-policy survival + override rate."""
    P.assert_passable()
    p = base.init()
    opt = base.Adam(p, C.LR)

    d = collect(p, 2500, seed=0, on_policy=False)  # warm start (expert drives)
    train_dagger(p, opt, d, epochs=250)
    print(
        f"[warm]   bare={base.survival(p, False, cap=10000)}  override_rate={override_rate(p):.3f}",
        flush=True,
    )

    for it in range(4):
        t = time.time()
        d = collect(p, 2500, seed=100 + it, on_policy=True)  # DAgger (net drives)
        train_dagger(p, opt, d, epochs=120)
        print(
            f"[dagger {it}] bare={base.survival(p, False, cap=10000)}  override_rate={override_rate(p):.3f}  ({time.time()-t:.0f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
