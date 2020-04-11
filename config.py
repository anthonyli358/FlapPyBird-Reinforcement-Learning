config = {'train': True,  # train or run the model
          'show_game': False,  # when training/evaluating it is much faster to not display the game graphics
          'max_score': 1000000,  # end the episode and update q-table when reaching this score
          'resume_score': 100000  # if dies above this score, resume training from this difficult segment
          }
