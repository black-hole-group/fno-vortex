Bleeding edge:
- Training data generated with Idefix instead of Fargo3d
- Implemented explicit train/val/test split: val is used for model selection
  and early stopping; test is final evaluation only (never seen during training)
- Given 5 frames, predict the next 20 ones
- Autoregressive rollout! Uses its own predictions to forecast the entire evolution
- Training saves a best-validation checkpoint (`model_best.pt`); inference prefers it automatically
- Current Idefix dataset: 50 simulations total (42 train / 6 val / 2 test)