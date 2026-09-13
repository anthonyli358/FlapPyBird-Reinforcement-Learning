"""
Expert Iteration for Flappy Bird

1. The shield is the main mechanic for the bird to not die
2. The policy net is distilled from the shield and is cheaper to run
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from . import dynamics as D
from . import planner as P
from .env import FlappyEnv
from .nets import ActorCritic, make_leaf_value, DEFAULT_POLICY
from .search import search


def expert_action(y: int, vel: int, pipes: list) -> int:
    """The combined action of the base center-seeking proposal then shield-corrected."""
    return P.shield(y, vel, pipes, P.center_seeking_action(y, vel, pipes))


class ReplayBuffer:
    """Fixed-capacity buffer that oversamples net-vs-expert disagreement states."""

    def __init__(self, cap: int = C.BUFFER_CAP):
        self.cap = cap
        self.x = np.zeros((cap, 5), dtype=np.float32)
        self.a = np.zeros(cap, dtype=np.int64)
        self.v = np.zeros(cap, dtype=np.float32)
        self.g = np.zeros(cap, dtype=np.bool_)  # net disagreed with expert here
        self.n = 0
        self.i = 0

    def add(self, x, a, v, disagree) -> None:
        """
        Store one labelled example, overwriting the oldest once full.

        Args:
            x: State feature vector.
            a: Expert (label) action, 0/1.
            v: Value target for the critic (from the search).
            disagree: True if the net's action differed from the expert here;
                these states are oversampled by `sample`.
        """
        self.x[self.i], self.a[self.i], self.v[self.i], self.g[self.i] = x, a, v, disagree
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch: int, priority_frac: float = C.PRIORITY_FRAC):
        """Sample half the batch from states with disagree, and half uniformly."""
        dis = np.where(self.g[:self.n])[0]
        n_pri = int(batch * priority_frac)
        if len(dis):
            pick_pri = np.random.choice(dis, size=n_pri, replace=len(dis) < n_pri)
        else:
            pick_pri = np.random.randint(0, self.n, size=n_pri)
        pick_uni = np.random.randint(0, self.n, size=batch - n_pri)
        idx = np.concatenate([pick_pri, pick_uni])
        return (torch.from_numpy(self.x[idx]), torch.from_numpy(self.a[idx]),
                torch.from_numpy(self.v[idx]))


def collect(net, buf: ReplayBuffer, frames: int, seed: int, on_policy: bool) -> None:
    """
    Roll out and label with the expert.

    Args:
        on_policy: If True the distillednet drives, if False the expert drives.
    """
    env = FlappyEnv(seed=seed)
    env.reset()
    leaf = make_leaf_value(net)
    for _ in range(frames):
        y, vel, pipes = env.state()
        label = expert_action(y, vel, pipes)
        net_a = net.act(y, vel, pipes)
        play = P.shield(y, vel, pipes, net_a) if on_policy else label
        value, _ = search(y, vel, pipes, C.SEARCH_DEPTH, leaf)  # value target for the critic
        buf.add(D.features(y, vel, pipes), label, value, net_a != label)
        _, _, done = env.step(bool(play))
        if done:  # only reachable if some gap is genuinely impassable
            env.reset()


def train(net, buf: ReplayBuffer, opt, epochs: int) -> tuple[float, float]:
    """Cross-entropy on the policy head, MSE on the value head."""
    lp = lv = 0.0
    steps = max(1, buf.n // C.BATCH)
    for _ in range(epochs):
        for _ in range(steps):
            x, a, v = buf.sample(C.BATCH)
            logits, value = net(x)
            l_pi = F.cross_entropy(logits, a)
            l_v = F.mse_loss(value, v)
            opt.zero_grad()
            (l_pi + C.VALUE_COEF * l_v).backward()
            opt.step()
            lp, lv = l_pi.item(), l_v.item()
    return lp, lv


@torch.no_grad()
def evaluate(net, seeds=range(5), cap: int = 20000) -> tuple[list, float]:
    """Bare-policy survival (no shield) and the deployed shield-override rate."""
    bare = []
    for s in seeds:
        env = FlappyEnv(seed=s); env.reset()
        for _ in range(cap):
            y, vel, pipes = env.state()
            _, _, done = env.step(bool(net.act(y, vel, pipes)))
            if done:
                bare.append(env.score); break
        else:
            bare.append(env.score)
    changed = total = 0
    for s in seeds:
        env = FlappyEnv(seed=s); env.reset()
        for _ in range(cap // 4):
            y, vel, pipes = env.state()
            na = net.act(y, vel, pipes)
            sa = P.shield(y, vel, pipes, na)
            changed += int(na != sa); total += 1
            env.step(bool(sa))
    return bare, changed / total


def train_policy() -> None:
    """
    Warm-start by behaviour cloning from the expert, then run on-policy DAgger rounds.
    The result is saved to the DEFAULT_POLICY.
    """
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    P.assert_passable()

    net = ActorCritic()
    opt = torch.optim.Adam(net.parameters(), lr=C.LR)
    buf = ReplayBuffer()

    collect(net, buf, C.WARM_FRAMES, seed=C.SEED, on_policy=False)
    lp, lv = train(net, buf, opt, C.WARM_EPOCHS)
    bare, orate = evaluate(net)
    print(f"[warm]     pi={lp:.3f} v={lv:.3f}  bare={bare}  override_rate={orate:.3f}")

    for it in range(C.EXIT_ITERS):
        collect(net, buf, C.EXIT_FRAMES, seed=C.SEED + 100 + it, on_policy=True)
        lp, lv = train(net, buf, opt, C.EXIT_EPOCHS)
        bare, orate = evaluate(net)
        print(f"[dagger {it}] pi={lp:.3f} v={lv:.3f}  bare={bare}  override_rate={orate:.3f}")

    os.makedirs(os.path.dirname(DEFAULT_POLICY), exist_ok=True)
    torch.save(net.state_dict(), DEFAULT_POLICY)
    print(f"saved {DEFAULT_POLICY}  (deploy as: shield.action(..., net.act))")


if __name__ == "__main__":
    train_policy()
