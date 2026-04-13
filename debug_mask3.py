import jax
from benchmarl.environments.lux.lux_env import LuxTorchRLEnv
import torch
import numpy as np

env = LuxTorchRLEnv(batch_size=10, max_steps=150, match_count=1)
td = env.reset()

for i in range(5):
    td["agents", "action"] = torch.zeros((10, 16), dtype=torch.long, device=env.device)
    td["action"] = td["agents", "action"]
    td = env.step(td).get("next")
    
mask = np.asarray(env.jax_obs["player_0"].units_mask)
print("Team IDs assigned:", env.team_ids.cpu().numpy())
print("Step 4 Team 0 mask sum:", mask[:, 0].sum(axis=1))
print("Step 4 Team 1 mask sum:", mask[:, 1].sum(axis=1))
