config = {
    'train': True,
    'train_type': 'q_table',  #dqn or q_table
    'show_game': False,
    'print_score': 10000,
    'max_score': 10000000,
    'resume_score': 100000,
    # Q-table
    'q_table_file': 'data/2026_q_values.json',
    'q_table_scores_file': 'data/2026_q_table_scores.json',
    # DQN
    'dqn_model_file': 'data/dqn_model.pt',
    'dqn_scores_file': 'data/dqn_scores.json',
}