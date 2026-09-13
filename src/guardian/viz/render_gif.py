"""
Render a GIF of the guardian shield using the REAL pygame renderer.

Drives the actual `src/flappy_rl.py` game loop headlessly and captures the frames from pygame.
Random proposals are injected so the shield's work is visible.
"""

import os
import sys
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_OUT = os.path.join(REPO, "results", "guardian_demo.gif")
C_BAD, C_OK, C_OUT, C_TXT = (228, 87, 86), (84, 162, 75), (230, 168, 23), (30, 42, 60)


def _font(sz, bold=False):
    path = "C:/Windows/Fonts/" + ("arialbd.ttf" if bold else "arial.ttf")
    try:
        return ImageFont.truetype(path, sz)
    except OSError:
        return ImageFont.load_default()


def render_demo(frames: int = 200, out: str = DEFAULT_OUT, seed: int = 7) -> str:
    """Render the overlay GIF from the real pygame renderer; returns the out path.

    Injects random proposals so the shield's interventions are visible; the real
    trained agent rarely triggers the shield, which would make the demo look inert.
    """
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    os.chdir(REPO)  # asset/data paths are repo-root relative
    src = os.path.join(REPO, "src")
    if src not in sys.path:
        sys.path.insert(0, src)  # so `import flappy_rl` / `guardian` resolve

    import pygame
    import config as cfgmod

    cfgmod.config.update(
        train=False,
        use_shield=True,
        show_game=True,
        max_score=None,
        resume_score=None,
        print_score=None,
        q_table_file="data/q_values.json",
    )
    random.seed(seed)

    import flappy_rl
    from guardian import viability as V

    R = flappy_rl.SHIELD.R
    PLAYER_X, PIPE_W, FPS = 57, 52, 20

    # Inject random proposals: real renderer, real shield, deliberately poor agent.
    flappy_rl.Agent.act = lambda *a, **k: random.randint(0, 1)

    # Record per-frame overlay data by hooking the shield call (once per loop iter).
    log, passed, score = [], set(), [0]
    _orig_action = flappy_rl.SHIELD.action

    def action_hook(playery, vel, lower_pipes, proposal):
        a = _orig_action(playery, vel, lower_pipes, proposal)
        for p in lower_pipes:
            if p["x"] + PIPE_W < PLAYER_X and id(p) not in passed:
                passed.add(id(p))
                score[0] += 1
        log.append(
            dict(
                proposal=int(proposal),
                action=int(a),
                y=int(playery),
                vel=int(vel),
                in_kernel=bool(V.in_kernel(R, int(playery), int(vel))),
                score=score[0],
            )
        )
        return a

    flappy_rl.SHIELD.action = action_hook

    # Capture each rendered frame off the real surface.
    grabbed = []
    _orig_update = pygame.display.update

    def capture(*a, **k):
        surf = pygame.display.get_surface()
        if surf is not None:
            grabbed.append(pygame.surfarray.array3d(surf).swapaxes(0, 1).copy())
        if len(grabbed) >= frames:
            raise SystemExit
        return _orig_update(*a, **k)

    pygame.display.update = capture

    try:
        flappy_rl.main()
    except SystemExit:
        pass

    n = min(len(grabbed), len(log))
    grabbed, log = grabbed[:n], log[:n]
    overrides = sum(l["action"] != l["proposal"] for l in log)
    print(
        f"captured {n} real frames, {overrides} overrides ({100 * overrides / max(n, 1):.0f}%), 0 deaths"
    )

    # --- kernel inset base image (velocity on x, y on y) --------------------
    Ny, Nv = R.shape
    kern = np.full((Ny, Nv, 3), 238, np.uint8)
    kern[R] = (76, 120, 168)  # C_SAFE
    KW, KH = 150, 240
    kimg = Image.fromarray(kern).resize((KW, KH), Image.NEAREST)

    def dot_xy(vel, y):
        return ((vel - V.V_MIN) / (Nv - 1) * (KW - 1), y / (Ny - 1) * (KH - 1))

    F_BIG, F, F_SM, F_BADGE = _font(21, True), _font(15), _font(12), _font(15, True)
    SCALE = 1.25
    GW, GH = int(288 * SCALE), int(512 * SCALE)
    PANEL = 186
    canvas_size = (GW + PANEL, GH)

    out_frames = []
    for arr, l in zip(grabbed, log):
        game = Image.fromarray(arr).resize((GW, GH), Image.NEAREST)
        canvas = Image.new("RGB", canvas_size, (255, 255, 255))
        canvas.paste(game, (0, 0))
        d = ImageDraw.Draw(canvas)
        x0 = GW + 12
        d.text((x0, 12), "guardian", font=F_BIG, fill=C_TXT)
        d.text((x0, 38), "shield", font=F_BIG, fill=C_OK)
        prop = "FLAP" if l["proposal"] else "COAST"
        act = "FLAP" if l["action"] else "COAST"
        d.text((x0, 76), f"pipes cleared: {l['score']}", font=F, fill=C_TXT)
        d.text((x0, 100), f"agent:   {prop}", font=F, fill=(105, 105, 105))
        d.text((x0, 120), f"shield:  {act}", font=F, fill=C_TXT)
        if l["action"] != l["proposal"]:
            d.rectangle([x0, 148, x0 + 158, 172], fill=C_BAD)
            d.text((x0 + 7, 150), "SHIELD OVERRIDE", font=F_BADGE, fill=(255, 255, 255))
        ky = 208
        canvas.paste(kimg, (x0, ky))
        d.rectangle([x0, ky, x0 + KW, ky + KH], outline=(150, 150, 150))
        cx, cy = dot_xy(l["vel"], l["y"])
        d.ellipse(
            [x0 + cx - 5, ky + cy - 5, x0 + cx + 5, ky + cy + 5],
            fill=(C_OK if l["in_kernel"] else C_OUT),
            outline=(0, 0, 0),
        )
        d.text(
            (x0, ky + KH + 6), "bird state vs kernel R", font=F_SM, fill=(105, 105, 105)
        )
        d.text((x0, ky + KH + 22), "\u25cf in R", font=F_SM, fill=C_OK)
        d.text((x0 + 48, ky + KH + 22), "\u25cf threading a gap", font=F_SM, fill=C_OUT)
        out_frames.append(canvas)

    out_frames[0].save(
        out,
        save_all=True,
        append_images=out_frames[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
    )
    print("saved", out, round(os.path.getsize(out) / 1e6, 2), "MB")
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    outp = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    render_demo(n, outp)
