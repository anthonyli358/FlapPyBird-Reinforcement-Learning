"""
Central configuration for the guardian shield.

All quantities being integers helps the shield be exact and lossless.
"""

try:
    import game_params as _gp
except ModuleNotFoundError:  # loaded as src.guardian.*
    from .. import game_params as _gp

# --- Physics (from the shared engine config) ---
GRAVITY = _gp.GRAVITY  # downward acceleration per frame (playerAccY)
MAX_VEL_Y = _gp.PLAYER_MAX_VEL_Y  # terminal fall speed (playerMaxVelY)
FLAP_ACC = _gp.FLAP_ACC  # velocity set on a flap; gravity is skipped that frame
PIPE_VEL_X = _gp.PIPE_VEL_X  # horizontal pipe scroll per frame (pipeVelX)

# --- Geometry (from the shared engine config) ---
SCREEN_W = _gp.SCREEN_WIDTH
SCREEN_H = _gp.SCREEN_HEIGHT
BASE_Y = int(
    _gp.BASE_Y
)  # 404: ground line (the exact death test is y+PLAYER_H >= BASE_Y-1)
PLAYER_X = _gp.PLAYER_X  # 57: bird's fixed x
PLAYER_W = _gp.PLAYER_W
PLAYER_H = _gp.PLAYER_H
PIPE_W = _gp.PIPE_W
PIPE_GAP = _gp.PIPE_GAP  # vertical opening between the pipe pair

# --- Geometry derived from the engine ----
PIPE_SPACING = _gp.SCREEN_WIDTH // 2  # 144: initial horizontal distance between pipes
_TOP_MIN = int(_gp.BASE_Y * 0.2)
_TOP_MAX = _TOP_MIN + int(_gp.BASE_Y * 0.6 - _gp.PIPE_GAP) - 1
GAP_CENTER_MIN = _TOP_MIN + _gp.PIPE_GAP // 2
GAP_CENTER_MAX = _TOP_MAX + _gp.PIPE_GAP // 2
PIPES_AHEAD = 4  # headless env.py for training/eval only
ENTRY_PX_MIN = 146  # x when a pipe becomes pipe_1 varies in a range
ENTRY_PX_MAX = 154

# --- Expert (center-seeking) ---
LOOKAHEAD = 1  # frames of coast simulated before deciding to flap
CENTER_OFFSET = 30  # aim below gap centre so a flap impulse does not overshoot

# --- Search / shield ---
SEARCH_DEPTH = 8  # frames of exact lookahead in the AlphaZero search
SHIELD_HORIZON = 70  # frames the reachability shield must keep survivable
CLEAR_SCALE = 100.0  # normaliser mapping pixel clearance -> ~[-1, 1]
DEAD_VALUE = -1.0  # leaf value for a collided state (death)

# --- Training ---
HIDDEN = 64
LR = 1e-3
VALUE_COEF = 1.0
WARM_FRAMES = 4000  # frames of center-seeking data for the warm start
WARM_EPOCHS = 300
EXIT_ITERS = 4  # expert-iteration rounds
EXIT_FRAMES = 4000  # frames collected per round
EXIT_EPOCHS = 150
BUFFER_CAP = 60000
PRIORITY_FRAC = 0.5  # fraction of each batch drawn from low-margin states
BATCH = 256
SEED = 0
