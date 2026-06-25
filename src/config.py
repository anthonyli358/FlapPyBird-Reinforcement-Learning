config = {
    'train': True,
    'train_type': 'q_table',  #dqn or q_table
    'show_game': False,
    'print_score': None,
    'max_score': 1e7,  # train for e.g. 10k episodes with max_score=10k, then 10, then validate at 10M
    'resume_score': 0,  # None to start, then e.g. 1000, None again when drilling early game
    # Q-table
    'q_table_file': 'data/2026_q_values.json',
    'q_table_scores_file': 'data/2026_q_table_scores.json',
    # DQN
    'dqn_model_file': 'data/dqn_model.pt',
    'dqn_scores_file': 'data/dqn_scores.json',
}