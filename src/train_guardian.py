from .guardian.viability import build_kernel
from .guardian.viz.results import main as make_figures
from .guardian.viz.render_gif import render_demo
from .guardian.numpy_verify import main as verify_numpy
from .guardian.dagger_verify import main as verify_dagger
from .guardian.expert_iteration import main as train_policy


def run():
    # --- Build the offline viability kernel for immortality ---
    build_kernel()

    # --- Regenerate the writeup figures ---
    # make_figures()          # all figures, or select "kernel" / "sweep" / "heatmap" / "survival"
    # render_demo()

    # --- Optional AlphaZero verification ---
    # verify_numpy()  # torch-free reproduction of the ExIt pipeline
    # verify_dagger()  # torch-free on-policy DAgger reproduction
    # train_policy()  # distil the net (needs torch)

    # Then play a real game using flappy_rl.py with use_shield=True in config
    pass


if __name__ == "__main__":
    run()
