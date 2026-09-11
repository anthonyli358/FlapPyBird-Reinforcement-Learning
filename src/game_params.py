"""
FlapPyBird engine parameters

Shared by flappy_rl.py and guardian/config.py. If parameters are changed
then guardian/viability.py should be reran and confirmed `|R| > 0` to
keep a working shield.
"""

# --- Screen / timing ---
FPS = 30
SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512
BASE_Y = SCREEN_HEIGHT * 0.79        # 404.48 ground line

# --- Bird / pipe geometry ---
PLAYER_X = int(SCREEN_WIDTH * 0.2)   # 57 — bird's fixed x
PLAYER_W = 34                        # bird sprite size (matches assets/sprites)
PLAYER_H = 24
PIPE_W = 52                          # pipe sprite width (matches assets/sprites)
PIPE_GAP = 100                       # vertical opening between the pipe pair

# --- Physics (per frame) ---
PIPE_VEL_X = -4                      # horizontal pipe scroll
GRAVITY = 1                          # downward acceleration (playerAccY)
PLAYER_MAX_VEL_Y = 10                # terminal fall speed
PLAYER_MIN_VEL_Y = -8               # (defined by the engine; unused by the RL loop)
FLAP_ACC = -9                        # velocity set on a flap
