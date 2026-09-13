"""
Torch-free NumPy reproduction of the expert-iteration (ExIt) distillation.
"""

import time
import numpy as np

from . import config as C
from . import dynamics as D
from . import planner as P
from .env import FlappyEnv
from .search import search

rng = np.random.default_rng(0)
H = 64


def init():
    """He-initialised (random starting weights) two-layer trunk with policy and value heads."""

    def w(a, b):
        return (rng.standard_normal((a, b)) * np.sqrt(2 / a)).astype(np.float64)

    return dict(
        W1=w(5, H),
        b1=np.zeros(H),
        W2=w(H, H),
        b2=np.zeros(H),
        Wp=w(H, 2),
        bp=np.zeros(2),
        Wv=w(H, 1),
        bv=np.zeros(1),
    )


def forward(p, x):
    """Forward pass which returns a cache of activations, policy logits, and tanh value."""
    z1 = x @ p["W1"] + p["b1"]
    h1 = np.maximum(z1, 0)
    z2 = h1 @ p["W2"] + p["b2"]
    h2 = np.maximum(z2, 0)
    logits = h2 @ p["Wp"] + p["bp"]
    value = np.tanh(h2 @ p["Wv"] + p["bv"])[:, 0]
    return dict(x=x, z1=z1, h1=h1, z2=z2, h2=h2, logits=logits, value=value)


def leaf_value_fn(p):
    """Return a `leaf(y, vel, pipes) -> float` scorer for thethat uses the value head for the search."""

    def lv(y, vel, pipes):
        return float(forward(p, D.features(y, vel, pipes)[None])["value"][0])

    return lv


def act(p, y, vel, pipes):
    """Greedy policy action (argmax of the policy logits) for one state."""
    return int(np.argmax(forward(p, D.features(y, vel, pipes)[None])["logits"][0]))


def grads(p, c, a, vtgt):
    """
    Manual backprop for one batch.
    
    Loss = cross-entropy(policy vs expert action) + VALUE_COEF * MSE.

    Returns:
      `(ce, mse, grads)`: the two scalar losses (for logging)
        and a dict of per-parameter gradients for the optimizer.
    """
    n = c["x"].shape[0]  # batch size

    # Policy head: softmax -> cross-entropy of the expert action `a`, and its gradient.
    sm = np.exp(c["logits"] - c["logits"].max(1, keepdims=True))
    sm /= sm.sum(1, keepdims=True)  # softmax probabilities
    ce = -np.log(sm[np.arange(n), a] + 1e-9).mean()  # cross-entropy loss
    dlogits = sm.copy()
    dlogits[np.arange(n), a] -= 1  # (softmax - onehot)...
    dlogits /= n  # ...averaged over the batch = dCE/dlogits

    # Value head: MSE vs target, gradient pushed back through the tanh.
    verr = c["value"] - vtgt  # prediction - target
    mse = (verr**2).mean()
    dval = (2 * verr / n) * (1 - c["value"] ** 2)  # dMSE/d(pre-tanh); tanh' = 1 - tanh^2

    # Head weights: grad = (that layer's input)^T @ (its output gradient).
    g = {}
    g["Wp"] = c["h2"].T @ dlogits  # policy head
    g["bp"] = dlogits.sum(0)
    g["Wv"] = c["h2"].T @ dval[:, None] * C.VALUE_COEF  # value head (weighted by VALUE_COEF)
    g["bv"] = np.array([dval.sum() * C.VALUE_COEF])

    # Shared trunk: combine both heads' gradients, then backprop through each
    # Linear + ReLU layer (ReLU passes gradient only where its input was > 0).
    dh2 = dlogits @ p["Wp"].T + (dval[:, None] @ p["Wv"].T) * C.VALUE_COEF  # grad into trunk output
    dz2 = dh2 * (c["z2"] > 0)  # ReLU backward, layer 2
    g["W2"] = c["h1"].T @ dz2
    g["b2"] = dz2.sum(0)
    dh1 = dz2 @ p["W2"].T
    dz1 = dh1 * (c["z1"] > 0)  # ReLU backward, layer 1
    g["W1"] = c["x"].T @ dz1
    g["b1"] = dz1.sum(0)
    return ce, mse, g


class Adam:
    """Minimal Adam optimizer over the parameter dict."""

    def __init__(self, p, lr=1e-3):
        self.lr = lr
        self.m = {k: np.zeros_like(v) for k, v in p.items()}
        self.v = {k: np.zeros_like(v) for k, v in p.items()}
        self.t = 0

    def step(self, p, g):
        """Apply one Adam update to `p` in place from grads `g`."""
        self.t += 1
        for k in p:
            self.m[k] = 0.9 * self.m[k] + 0.1 * g[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * g[k] ** 2
            mh = self.m[k] / (1 - 0.9**self.t)
            vh = self.v[k] / (1 - 0.999**self.t)
            p[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def collect(p, frames, seed, warm):
    """
    Roll out and collect (features, shielded-action label, search value, clearance).

    warm=True: center-seeking drives the rollout (behaviour-cloning data).
    warm=False: the depth-8 search (bootstrapped by the net's value head) drives.
    """
    X, A, V, M = [], [], [], []
    env = FlappyEnv(seed=seed)
    env.reset()
    leaf = None if warm else leaf_value_fn(p)
    for _ in range(frames):
        y, vel, pipes = env.state()
        value, s_a = search(y, vel, pipes, C.SEARCH_DEPTH, leaf)
        proposal = P.center_seeking_action(y, vel, pipes) if warm else s_a
        a = P.shield(y, vel, pipes, proposal)  # label the safe expert actually played
        X.append(D.features(y, vel, pipes))
        A.append(a)
        V.append(value)
        M.append(D.clearance(y, pipes))
        _, _, done = env.step(bool(a))
        if done:
            env.reset()
    return (
        np.array(X, np.float64),
        np.array(A),
        np.array(V, np.float64),
        np.array(M, np.float64),
    )


def train(p, opt, data, epochs, batch=256):
    """Train both heads on the collected data, oversampling low-clearance (risky) states."""
    X, A, V, M = data
    n = len(X)
    order_low = np.argsort(M)  # low-margin first for priority
    for _ in range(epochs):
        for _ in range(max(1, n // batch)):
            npri = batch // 2
            pri = np.random.choice(order_low[: max(npri * 4, 1)], npri)
            uni = np.random.randint(0, n, batch - npri)
            idx = np.concatenate([pri, uni])
            c = forward(p, X[idx])
            ce, mse, g = grads(p, c, A[idx], V[idx])
            opt.step(p, g)
    return ce, mse


def survival(p, use_shield, seeds=range(5), cap=20000):
    """Pipes cleared per seed by the bare net (or net+shield if use_shield."""
    out = []
    for s in seeds:
        env = FlappyEnv(seed=s)
        env.reset()
        for f in range(cap):
            y, vel, pipes = env.state()
            a = act(p, y, vel, pipes)
            if use_shield:
                a = P.shield(y, vel, pipes, a)
            _, _, done = env.step(bool(a))
            if done:
                out.append(env.score)
                break
        else:
            out.append(env.score)
    return out


def main():
    """Reproduce the ExIt pipeline in NumPy and print bare-policy survival."""
    P.assert_passable()
    p = init()
    opt = Adam(p, C.LR)

    # baseline: shallow search alone, no learning
    base = []
    for s in range(5):
        env = FlappyEnv(seed=s)
        env.reset()
        for f in range(20000):
            y, vel, pipes = env.state()
            _, _, done = env.step(bool(search(y, vel, pipes, C.SEARCH_DEPTH)[1]))
            if done:
                base.append(env.score)
                break
        else:
            base.append(env.score)
    print("depth-8 search alone (no net):   scores", base)

    t = time.time()
    d = collect(p, 2500, seed=0, warm=True)
    ce, mse = train(p, opt, d, epochs=250)
    print(
        f"[warm]   ce={ce:.3f} mse={mse:.3f}  bare_policy scores={survival(p, False)}  ({time.time()-t:.0f}s)"
    )

    for it in range(3):
        t = time.time()
        d = collect(p, 2500, seed=100 + it, warm=False)
        ce, mse = train(p, opt, d, epochs=150)
        print(
            f"[exit {it}] ce={ce:.3f} mse={mse:.3f}  bare_policy scores={survival(p, False)}  ({time.time()-t:.0f}s)"
        )

    print(
        "bare policy is immortal once coupled with the shield",
        survival(p, True, seeds=range(3), cap=8000),
    )


if __name__ == "__main__":
    main()
