config = {
    "train": False,  # False: RUN mode, True: train the agent
    "train_type": "q_table",  # dqn or q_table
    "use_shield": True,  # use the guardian safety shield (RUN mode only)
    "show_game": True,
    "print_score": None,
    "max_score": 1e8,  # None = run forever (immortal); a number caps the run
    "resume_score": None,  # None to start, then e.g. 1000, None again when drilling early game

    # Q-tables (LOAD and SAVE are separate files)
    "q_table_file": "data/q_values_resume.json",  # LOAD: base/resume table (read-only)
    "q_table_scores_file": "data/training_values_resume.json",  # LOAD: curves for the current q-table
    "q_table_save_file": "data/q_values_train.json",  # SAVE: training writes and continues from here
    "q_table_scores_save_file": "data/training_values_train.json",  # SAVE: training curves

    # DQN
    "dqn_model_file": "data/dqn_model.pt",
    "dqn_scores_file": "data/dqn_scores.json",
}
