import jax
from benchmarl.environments.lux.lux_env import LuxTorchRLEnv
import torch
import numpy as np

env = LuxTorchRLEnv(batch_size=1, max_steps=150, match_count=1)
td = env.reset()

for i in range(5):
    td["agents", "action"] = torch.zeros((1, 16), dtype=torch.long, device=env.device)
    td["action"] = td["agents", "action"]
    td = env.step(td).get("next")
    
    mask = np.asarray(env.jax_obs["player_0"].units_mask)
    print(f"Step {i}:")
    print(f"  Team 0 mask sum: {mask[0, 0].sum()}")
    print(f"  Team 1 mask sum: {mask[0, 1].sum()}")
