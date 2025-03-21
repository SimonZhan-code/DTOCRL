## Environment
Original BeliefEncoder Environment should be perfectly fine for now, later might need update for Minari for other environments
## Training
To run the experiment use the following command
```
python new_scripts/train_dtocrl.py
```
## Config
Details configs are list below
```
		"env_name": ["hopper-medium-v2"],
        "device":  ["cuda" if torch.cuda.is_available() else "cpu"],
        "seed": [0],  # Sets Gym, PyTorch and Numpy seeds
        "gamma": [0.99],
        "alpha": [0.2],
        "total_step": [int(1e6)],  # Max time steps to run environment
        "buffer_size": [int(1e6)],  # Replay buffer size
        "batch_size": [256],  # Batch size for all networks
        "lr_actor": [3e-4],
        "lr_critic": [3e-4],
        "lr_dynamic": [3e-4],
        "lr_alpha": [1e-3],
        "latent_dim": [256],
        "num_layers": [10],
        "num_heads": [4],
        "attention_dropout": [0.1],
        "residual_dropout": [0.1],
        "hidden_dropout": [0.1],
        "soft_update_factor": [5e-3],
        "learn_start": [int(1e2)],
        "rollout_freq": [int(1e3)],
        "num_rollout": [int(1e3)],
        "evaluate_freq": [int(1e4)],
        "num_evaluation": [10],
        "delay": [int(8)],
        "dynamic_type": ['trans'],
        "use_automatic_entropy_tuning": [False],
        "cql_n_actions": [10],
        "cql_importance_sample": [True],
        "cql_lagrange": [False],
        "cql_target_action_gap": [-1.0],
        "lagrange_threshold": [10.0],
        "cql_temp": [1.0],
        "cql_alpha": [0.2],
        "wandb": [True],
        "backup_entropy": [False],
        "cql_max_target_backup": [False],
        "cql_weight": [1.0],
        "fake_real_ratio": [1.0],
        "dynamic_train_threshold": [2e5],
```
Please add environments need to train to `env_name` part, and also various seeds and delays in `seed` and `delay` accordingly. `rollout_freq`, `num_rollout`, and `learn_start` should be tuned accordingly, depending on how well the latent dynamic model is trained. 