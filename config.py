config = {'train': True,  # train or run the model
          'show_game': False,  # when training/evaluating it is much faster to not display the game graphics
          'print_score': 10000,  # print when a multiple of this score is reached
          'max_score': None,  # end the episode and update q-table when reaching this score
          'resume_score': None,  # if dies above this score, resume training from this difficult segment
          }
