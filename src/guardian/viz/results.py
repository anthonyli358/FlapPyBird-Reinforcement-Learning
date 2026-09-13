"""
Generate results graphs
"""

from __future__ import annotations

import os
import random

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .. import config as C
from .. import viability as V
from .. import realtime as RT

# --- engine constants (match flappy_rl.py) ----------------------------------
SCREENWIDTH, SCREENHEIGHT = 288, 512
PIPEGAPSIZE = 100
BASEY = SCREENHEIGHT * 0.79  # 404.48
PLAYER_H, PLAYER_W, PLAYER_X = 24, 34, 57
PIPE_W, PIPE_H = 52, 320

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "results"))

# Brand-neutral, colour-blind-safe palette
C_SAFE = "#4C78A8"  # kernel / safe
C_OK = "#54A24B"  # survives / gap
C_BAD = "#E45756"  # dies / override
C_OUT = "#E6A817"  # amber / threading a gap
C_NEUT = "#B0B0B0"


def _pipes_from(lower_pipes):
    """Adapter: on-screen lower pipes -> planning pipes (px, gap_center)."""
    return [
        (p["x"], p["y"] - PIPEGAPSIZE // 2) for p in lower_pipes if p["x"] < SCREENWIDTH
    ]


def _grp(rng):
    gapY = rng.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE)) + int(BASEY * 0.2)
    return {"upper_y": gapY - PIPE_H, "lower_y": gapY + PIPEGAPSIZE}


def _collides(y, lower_pipes):
    if y + PLAYER_H >= BASEY - 1:
        return True
    if y < 0:
        return True
    for p in lower_pipes:
        if PLAYER_X + PLAYER_W > p["x"] and PLAYER_X < p["x"] + PIPE_W:
            if y < p["y"] - PIPEGAPSIZE or y + PLAYER_H > p["y"]:
                return True
    return False


def simulate(R, mode, frames, seed, record=False, on_step=None, propose=None):
    """Run the real game under a proposal policy, shielded by kernel ``R``.

    Args:
        on_step: Optional `on_step(y, vel, proposal, action)` called each
            frame, to tally the override heatmap without storing a trajectory.
        propose: Optional `propose(playerx, playery, playerVelY, lowerPipes)
            -> 0|1` supplying the proposal when `mode == "trained"` (e.g. a
            trained Q-agent's action).
        record: When True, `traj` holds a per-frame trajectory list for the how-it-works figure.

    Returns:
        dict: Run summary with keys `survived`, `died_at`, `score`,
        `overrides` and `frames_run`.
    """
    rng = random.Random(seed)
    prng = random.Random(seed ^ 0xABCDEF)
    playery = int((SCREENHEIGHT - PLAYER_H) / 2)
    playerVelY = -9
    p1, p2 = _grp(rng), _grp(rng)
    upperPipes = [
        {"x": SCREENWIDTH + 200, "y": p1["upper_y"]},
        {"x": SCREENWIDTH + 200 + (SCREENWIDTH // 2), "y": p2["upper_y"]},
    ]
    lowerPipes = [
        {"x": SCREENWIDTH + 200, "y": p1["lower_y"]},
        {"x": SCREENWIDTH + 200 + (SCREENWIDTH // 2), "y": p2["lower_y"]},
    ]

    overrides = 0
    score = 0
    traj = []
    world0 = 0  # cumulative scroll, for world coordinates
    for f in range(frames):
        if mode == "flap":
            proposal = 1
        elif mode == "noflap":
            proposal = 0
        elif mode == "random":
            proposal = prng.randint(0, 1)
        elif mode == "trained":
            proposal = 1 if propose(PLAYER_X, playery, playerVelY, lowerPipes) else 0
        else:  # adversary: push toward the far gap extreme
            proposal = 1 if playery + PLAYER_H / 2 > 271 else 0

        pipes = _pipes_from(lowerPipes)
        action = RT.robust_action(playery, playerVelY, pipes, proposal, R)
        overrode = action != proposal
        overrides += overrode
        if on_step is not None:
            on_step(playery, playerVelY, proposal, action)

        if record:
            gov = RT._governing(pipes)
            traj.append(
                {
                    "f": f,
                    "world_x": PLAYER_X + world0,
                    "y": playery,
                    "vel": playerVelY,
                    "action": action,
                    "proposal": proposal,
                    "override": overrode,
                    "in_kernel": bool(V.in_kernel(R, playery, playerVelY)),
                    "pipes": [
                        (p["x"] + world0, p["y"] - PIPEGAPSIZE, p["y"])
                        for p in lowerPipes
                    ],
                    "screen_pipes": [
                        (p["x"], p["y"] - PIPEGAPSIZE, p["y"]) for p in lowerPipes
                    ],
                    "gov": gov[0] if gov else None,
                    "score": score,
                }
            )

        flapped = False
        if action == 1:
            playerVelY = -9
            flapped = True
        if _collides(playery, lowerPipes):
            return {
                "survived": False,
                "died_at": f,
                "score": score,
                "overrides": overrides,
                "frames_run": f,
                "traj": traj,
            }
        # score: bird centre crosses a pipe centre
        mid = PLAYER_X + PLAYER_W / 2
        for p in upperPipes:
            pm = p["x"] + PIPE_W / 2
            if pm <= mid < pm + 4:
                score += 1
        if playerVelY < 10 and not flapped:
            playerVelY += 1
        playery += int(min(playerVelY, BASEY - playery - PLAYER_H))
        for u, l in zip(upperPipes, lowerPipes):
            u["x"] += -4
            l["x"] += -4
        world0 += 4
        if 0 < upperPipes[0]["x"] < 5:
            n = _grp(rng)
            upperPipes.append({"x": SCREENWIDTH + 10, "y": n["upper_y"]})
            lowerPipes.append({"x": SCREENWIDTH + 10, "y": n["lower_y"]})
        if upperPipes[0]["x"] < -PIPE_W:
            upperPipes.pop(0)
            lowerPipes.pop(0)

    return {
        "survived": True,
        "died_at": None,
        "score": score,
        "overrides": overrides,
        "frames_run": frames,
        "traj": traj,
    }


# ---------------------------------------------------------------------------
# Figure 1: the viability kernel R
# ---------------------------------------------------------------------------
def fig_kernel(R):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    vmin = V.V_MIN
    extent = [
        vmin - 0.5,
        V.V_MAX + 0.5,
        V.Y_MAX + 0.5,
        -0.5,
    ]  # y inverted (down = down)
    ax.imshow(
        R.astype(float),
        aspect="auto",
        extent=extent,
        cmap=matplotlib.colors.ListedColormap(["#EEEEEE", C_SAFE]),
        interpolation="nearest",
    )
    ax.axhspan(
        C.GAP_CENTER_MIN, C.GAP_CENTER_MAX, xmin=0, xmax=1, color=C_OK, alpha=0.10, lw=0
    )
    ax.axhline(C.GAP_CENTER_MIN, color=C_OK, lw=1, ls="--", alpha=0.7)
    ax.axhline(C.GAP_CENTER_MAX, color=C_OK, lw=1, ls="--", alpha=0.7)
    ax.text(
        V.V_MAX - 0.3,
        (C.GAP_CENTER_MIN + C.GAP_CENTER_MAX) / 2,
        "gap-centre band",
        color=C_OK,
        ha="right",
        va="center",
        fontsize=9,
    )
    ys, _ = np.where(R)
    ax.set_title(
        f"Viability kernel R  (|R| = {int(R.sum())} / {R.size} states, "
        f"y ∈ [{ys.min()}, {ys.max()}])",
        fontsize=11,
    )
    ax.set_xlabel("bird velocity (px/frame)")
    ax.set_ylabel("bird y, top-left (px, down ↓)")
    ax.set_xticks(range(vmin, V.V_MAX + 1, 2))
    handles = [
        Rectangle((0, 0), 1, 1, color=C_SAFE),
        Rectangle((0, 0), 1, 1, color="#EEEEEE"),
    ]
    ax.legend(
        handles,
        [
            "safe (in R): can thread any next gap and return to R",
            "unsafe: some gap forces a death",
        ],
        loc="lower left",
        fontsize=8,
        framealpha=0.95,
    )
    fig.tight_layout()
    out = os.path.join(RESULTS, "guardian_kernel.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2: override rate by proposal policy (all immortal)
# ---------------------------------------------------------------------------
def fig_overrides(R, frames, seeds):
    modes = ["noflap", "adversary", "random", "flap"]
    labels = {
        "noflap": "always\nno-flap",
        "adversary": "adversary",
        "random": "random",
        "flap": "always\nflap",
    }
    rates, deaths = [], []
    for m in modes:
        rs, dz = [], 0
        for s in seeds:
            r = simulate(R, m, frames, s)
            rs.append(100 * r["overrides"] / r["frames_run"])
            dz += not r["survived"]
        rates.append(np.mean(rs))
        deaths.append(dz)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar([labels[m] for m in modes], rates, color=C_SAFE, width=0.62)
    for b, r, d in zip(bars, rates, deaths):
        ax.text(
            b.get_x() + b.get_width() / 2,
            r + 1.5,
            f"{r:.0f}%",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
        ax.text(
            b.get_x() + b.get_width() / 2,
            3,
            "0 deaths" if d == 0 else f"{d} deaths",
            ha="center",
            fontsize=8,
            color="white" if r > 12 else C_OK,
            fontweight="bold",
        )
    ax.set_ylim(0, 100)
    ax.set_ylabel("frames the shield overrode the proposal (%)")
    ax.set_title(
        f"The worse the proposal, the more the shield steps in — "
        f"survival is unconditional\n"
        f"({len(seeds)} seeds × {frames:,} frames each)",
        fontsize=10,
    )
    fig.tight_layout()
    out = os.path.join(RESULTS, "guardian_override_rates.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out, dict(zip(modes, rates))


# ---------------------------------------------------------------------------
# Figure 3: the phase-robustness fix (single-phase kernel vs phase-robust)
# ---------------------------------------------------------------------------
def fig_phase_fix(R_robust, frames, seeds, single_cap=60000):
    print("  building single-phase (idealised px=149) kernel for the before/after ...")
    R_single = V.compute_kernel(entries=[149])
    single_v, n_died, robust_ok = [], 0, 0
    for s in seeds:
        r = simulate(R_single, "flap", single_cap, s)  # always-flap = hardest proposal
        if not r["survived"]:
            single_v.append(r["score"])  # pipes cleared before death
            n_died += 1
        robust_ok += simulate(R_robust, "flap", frames, s)["survived"]

    top = max(single_v) * 1.4
    fig, ax = plt.subplots(figsize=(7, 4.4))
    # left: single-phase kernel dies
    ax.bar(0, np.mean(single_v), color=C_BAD, width=0.5, zorder=2)
    ax.scatter(np.zeros(len(single_v)), single_v, color="#7a1f1f", zorder=5, s=26)
    for v in single_v:
        ax.annotate(
            f"{v}",
            (0, v),
            textcoords="offset points",
            xytext=(11, 0),
            fontsize=8,
            va="center",
            color="#7a1f1f",
        )
    ax.annotate(
        f"{n_died}/{len(seeds)} seeds die\n(median {int(np.median(single_v))} pipes)",
        (0, np.mean(single_v)),
        textcoords="offset points",
        xytext=(0, 30),
        ha="center",
        fontsize=8.5,
        color=C_BAD,
        fontweight="bold",
    )
    # right: phase-robust kernel is immortal -> bar runs off the top
    ax.bar(
        1,
        top,
        color=C_OK,
        width=0.5,
        hatch="//",
        edgecolor="#2f6b28",
        alpha=0.85,
        zorder=2,
    )
    ax.annotate(
        "",
        xy=(1, top * 1.10),
        xytext=(1, top * 0.98),
        arrowprops=dict(arrowstyle="-|>", color="#2f6b28", lw=2),
    )
    ax.annotate(
        "immortal — ∞\n0 deaths\n(6M-frame soak)",
        (1, top * 0.5),
        ha="center",
        va="center",
        fontsize=10,
        color="white",
        fontweight="bold",
    )

    ax.set_ylim(0, top * 1.18)
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [
            "single-phase kernel\n(idealised px = 149)",
            "phase-robust kernel\n(px ∈ [146, 154])",
        ]
    )
    ax.set_ylabel(
        f"pipes cleared before death\n(always-flap; single-phase run to {single_cap:,} frames)"
    )
    ax.set_title(
        "Why phase-robustness matters: the idealised-cadence kernel is\n"
        "immortal on the sandbox but DIES in the real game",
        fontsize=10,
    )
    fig.tight_layout()
    out = os.path.join(RESULTS, "guardian_phase_fix.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out, single_v, robust_ok


# ---------------------------------------------------------------------------
# Figure 4: how it works — a shielded run through the pipes
# ---------------------------------------------------------------------------
def fig_trajectory(R, seed=3, frames=430):
    r = simulate(R, "random", frames, seed, record=True)
    traj = r["traj"]
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(9, 5.6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    # draw each pipe once (dedupe by rounded world x of its top)
    seen = set()
    for t in traj:
        for wx, gap_top, gap_bot in t["pipes"]:
            key = round(wx / 4)
            if key in seen:
                continue
            seen.add(key)
            ax.add_patch(
                Rectangle((wx, -5), PIPE_W, gap_top + 5, color=C_OK, alpha=0.30, lw=0)
            )
            ax.add_patch(
                Rectangle(
                    (wx, gap_bot), PIPE_W, BASEY - gap_bot, color=C_OK, alpha=0.30, lw=0
                )
            )
    xs = [t["world_x"] for t in traj]
    ys = [t["y"] for t in traj]
    ax.plot(xs, ys, color=C_SAFE, lw=1.6, label="bird path", zorder=4)
    ov = [(t["world_x"], t["y"]) for t in traj if t["override"]]
    if ov:
        ax.scatter(
            [p[0] for p in ov],
            [p[1] for p in ov],
            color=C_BAD,
            s=12,
            zorder=5,
            label="shield overrode the proposal",
        )
    ax.axhline(BASEY, color="#8a6d3b", lw=2)
    ax.text(xs[0], BASEY - 4, "ground", color="#8a6d3b", fontsize=8, va="bottom")
    ax.set_ylim(BASEY + 10, -10)  # inverted
    ax.set_ylabel("bird y (px, down ↓)")
    ax.set_title(
        f"How it works: a shielded run under a RANDOM proposal "
        f"(seed {seed}) — {int(100*r['overrides']/r['frames_run'])}% overridden, "
        f"0 deaths",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    # velocity panel
    vel = [t["vel"] for t in traj]
    ax2.plot(xs, vel, color=C_NEUT, lw=1.2)
    flaps = [(t["world_x"], t["vel"]) for t in traj if t["action"] == 1]
    ax2.scatter(
        [p[0] for p in flaps], [p[1] for p in flaps], color=C_SAFE, s=8, label="flap"
    )
    ax2.axhline(0, color="#dddddd", lw=0.8)
    ax2.set_ylabel("velocity")
    ax2.set_xlabel("world x (px) — the bird moves left → right through the pipes")
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = os.path.join(RESULTS, "guardian_trajectory.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_override_heatmap(R, frames, seeds):
    Ny, NV = R.shape
    visits = np.zeros((Ny, NV))
    over = np.zeros((Ny, NV))

    def tally(y, vel, prop, act):
        if 0 <= y < Ny and V.V_MIN <= vel <= V.V_MAX:
            visits[y, vel - V.V_MIN] += 1
            over[y, vel - V.V_MIN] += act != prop

    for s in seeds:
        simulate(R, "random", frames, s, on_step=tally)

    # bin y (integer dynamics visit sparse y per velocity -> smooth the striping)
    BIN = 8
    nb = (Ny + BIN - 1) // BIN
    vb, ob = np.zeros((nb, NV)), np.zeros((nb, NV))
    for r in range(Ny):
        vb[r // BIN] += visits[r]
        ob[r // BIN] += over[r]
    rate = np.divide(ob, vb, out=np.full_like(ob, np.nan), where=vb > 0)

    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.set_facecolor("#F5F5F5")
    extent = [V.V_MIN - 0.5, V.V_MAX + 0.5, V.Y_MAX + 0.5, -0.5]
    im = ax.imshow(
        rate,
        aspect="auto",
        extent=extent,
        cmap="magma_r",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax.contour(
        np.arange(V.V_MIN, V.V_MAX + 1),
        np.arange(Ny),
        R.astype(float),
        levels=[0.5],
        colors=C_SAFE,
        linewidths=1.8,
    )
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("shield override rate in this state")
    ax.set_title(
        "Where the shield actually intervenes (random proposals)\n"
        "blue line = kernel R boundary · grey = states never visited",
        fontsize=10,
    )
    ax.set_xlabel("bird velocity (px/frame)")
    ax.set_ylabel("bird y, top-left (px, down ↓)")
    ax.set_xticks(range(V.V_MIN, V.V_MAX + 1, 2))
    fig.tight_layout()
    out = os.path.join(RESULTS, "guardian_override_heatmap.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_difficulty_sweep(gaps=range(45, 141, 5)):
    """|R| vs pipe-gap size: the safety phase transition."""
    gaps = list(gaps)
    saved = (C.PIPE_GAP, C.GAP_CENTER_MIN, C.GAP_CENTER_MAX, V.HALF)
    sizes = []
    try:
        for gs in gaps:
            top_min = int(BASEY * 0.2)
            top_max = top_min + int(BASEY * 0.6 - gs) - 1  # getRandomPipe range
            C.PIPE_GAP = gs
            C.GAP_CENTER_MIN = top_min + gs // 2
            C.GAP_CENTER_MAX = top_max + gs // 2
            V.HALF = gs // 2
            sizes.append(int(V.compute_kernel().sum()))
    finally:
        C.PIPE_GAP, C.GAP_CENTER_MIN, C.GAP_CENTER_MAX, V.HALF = saved
    sizes = np.array(sizes)

    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(gaps, sizes, "-o", color=C_SAFE, ms=4)
    ax.fill_between(gaps, 0, sizes, color=C_SAFE, alpha=0.12)
    # actual game gap
    if 100 in gaps:
        i = gaps.index(100)
        ax.scatter([100], [sizes[i]], color=C_OK, zorder=6, s=60)
        ax.annotate(
            f"this game\n(gap 100 → |R|={sizes[i]})",
            (100, sizes[i]),
            textcoords="offset points",
            xytext=(-8, 14),
            fontsize=8.5,
            color="#2f6b28",
            fontweight="bold",
            ha="right",
        )
    # collapse point
    zero = [g for g, s in zip(gaps, sizes) if s == 0]
    if zero:
        gz = max(zero)
        ax.axvspan(gaps[0], gz + 0.5, color=C_BAD, alpha=0.08)
        ax.annotate(
            f"|R| = 0 for gap ≤ {gz}px:\nno controller can be immortal",
            (gz, 0),
            textcoords="offset points",
            xytext=(8, 40),
            fontsize=8.5,
            color=C_BAD,
        )
    ax.set_xlabel("pipe-gap size (px)")
    ax.set_ylabel("viability kernel size |R| (states)")
    ax.set_title(
        "Difficulty sweep: how much safety margin the game leaves\n"
        "(smaller gap ⇒ smaller kernel ⇒ eventually immortality is impossible)",
        fontsize=10,
    )
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    out = os.path.join(RESULTS, "guardian_difficulty_sweep.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out, list(zip(gaps, sizes.tolist()))


def fig_latency(R, frames, seed):
    """Per-frame robust_action latency distribution vs the real-time budget."""
    import time

    times = []
    orig = RT.robust_action

    def timed(*a, **k):
        t0 = time.perf_counter()
        r = orig(*a, **k)
        times.append((time.perf_counter() - t0) * 1000)
        return r

    RT.robust_action = timed
    try:
        simulate(R, "random", frames, seed)
    finally:
        RT.robust_action = orig
    t = np.array(times)
    med, p99, mx = np.median(t), np.percentile(t, 99), t.max()
    budget = 1000 / 30  # 33.3 ms at 30 FPS

    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.hist(t, bins=60, color=C_SAFE, alpha=0.9)
    for val, lab, col in [
        (med, f"median {med:.2f} ms", "#333333"),
        (p99, f"p99 {p99:.2f} ms", C_OUT),
        (mx, f"max {mx:.2f} ms", C_BAD),
    ]:
        ax.axvline(val, color=col, ls="--", lw=1.4)
        ax.text(
            val,
            ax.get_ylim()[1] * 0.9,
            "  " + lab,
            color=col,
            fontsize=8.5,
            rotation=90,
            va="top",
        )
    ax.set_xlabel("shield decision time per frame (ms)")
    ax.set_ylabel(f"frames ({len(t):,} total)")
    ax.set_title(
        f"Deployed cost: {med:.2f} ms median, {mx:.2f} ms worst case —\n"
        f"vs a {budget:.0f} ms/frame budget at 30 FPS "
        f"({budget / mx:.0f}× headroom even at the worst frame)",
        fontsize=10,
    )
    fig.tight_layout()
    out = os.path.join(RESULTS, "guardian_latency.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out, dict(median=med, p99=p99, max=mx)


def fig_survival():
    """Bare trained agent (dies) vs guardian-shielded (never dies)."""
    import json

    path = os.path.join(_HERE, "..", "..", "..", "data", "validation_resume.json")
    scores = np.array(sorted(json.load(open(path))["scores"]))
    n = len(scores)
    surv = np.arange(n, 0, -1) / n  # fraction of runs reaching >= score

    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.step(
        scores,
        surv,
        where="post",
        color=C_BAD,
        lw=2,
        label=f"bare Q-agent ({n} runs which die, max 6.72M score)",
    )
    ax.scatter(scores, surv, color=C_BAD, s=14, zorder=5)
    ax.hlines(
        1.0,
        scores.min(),
        2e7,
        color=C_OK,
        lw=2.6,
        label="guardian-shielded (immortal, 6M frames tested)",
    )
    ax.annotate(
        "→ ∞",
        (1.4e7, 1.0),
        color="#2f6b28",
        fontweight="bold",
        fontsize=11,
        va="center",
    )
    ax.set_xscale("log")
    ax.set_xlim(scores.min() * 0.7, 2e7)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("pipes cleared in a run (log scale)")
    ax.set_ylabel("fraction of runs still alive")
    ax.set_title(f"The guardian makes the bird never die", fontsize=9.5)
    ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    out = os.path.join(RESULTS, "guardian_survival.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main(which="all"):
    os.makedirs(RESULTS, exist_ok=True)
    R = np.load(RT.DEFAULT_KERNEL)
    print(f"kernel |R| = {int(R.sum())} / {R.size}")

    seeds = list(range(5))
    F = 20000

    if which in ("all", "kernel"):
        print("Fig 1: kernel heatmap ...")
        print("  ->", fig_kernel(R))
    if which in ("all", "overrides"):
        print("Fig 2: override rates ...")
        p2, rates = fig_overrides(R, F, seeds)
        print("  ->", p2, rates)
    if which in ("all", "phase"):
        print("Fig 3: phase-fix before/after ...")
        p3, single_v, robust_ok = fig_phase_fix(R, F, seeds)
        print("  ->", p3)
        print(f"     single-phase kernel, pipes-before-death by seed: {single_v}")
        print(
            f"     phase-robust kernel: {robust_ok}/{len(seeds)} seeds immortal (capped)"
        )
    if which in ("all", "trajectory"):
        print("Fig 4: how-it-works trajectory ...")
        print("  ->", fig_trajectory(R))
    if which in ("all", "heatmap"):
        print("Fig 5: override heatmap ...")
        print("  ->", fig_override_heatmap(R, 15000, list(range(3))))
    if which in ("all", "sweep"):
        print("Fig 6: difficulty sweep |R| vs gap ...")
        p, data = fig_difficulty_sweep()
        print("  ->", p)
        print("     (gap, |R|):", data)
    if which in ("all", "latency"):
        print("Fig 7: latency histogram ...")
        p, stats = fig_latency(R, 12000, 1)
        print("  ->", p, stats)
    if which in ("all", "survival"):
        print("Fig 8: bare vs shielded survival ...")
        print("  ->", fig_survival())
    print(
        "done. (for the animated GIF from the real renderer: "
        "python src/guardian/render_gif.py)"
    )


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "all")
