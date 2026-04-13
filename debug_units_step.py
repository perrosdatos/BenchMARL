import jax
from benchmarl.environments.lux.lux_env import LuxTorchRLEnv
import torch
env = LuxTorchRLEnv(batch_size=1, max_steps=150, match_count=1)
td = env.reset()
import numpy as np

for i in range(10):
    td["agents", "action"] = torch.zeros((1, 16), dtype=torch.long, device=env.device)
    td["action"] = td["agents", "action"]
    td = env.step(td).get("next")
    
    mask = np.asarray(env.jax_obs["player_0"].units_mask)
    if mask.any():
        print(f"Units spawned at step {i}!")
        break
    else:
        print(f"Step {i}: no units")
