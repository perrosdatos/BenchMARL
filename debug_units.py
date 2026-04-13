import jax
from benchmarl.environments.lux.lux_env import LuxTorchRLEnv
env = LuxTorchRLEnv(batch_size=1, max_steps=150, match_count=1)
env.reset()
import numpy as np
mask = np.asarray(env.jax_obs["player_0"].units_mask)
print("Units mask shape:", mask.shape)
print("Any true?", mask.any())
print("True elements:", np.where(mask))
