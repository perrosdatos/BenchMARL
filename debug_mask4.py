import jax
from benchmarl.environments.lux.lux_env import LuxTorchRLEnv
import numpy as np

env = LuxTorchRLEnv(batch_size=1, max_steps=150, match_count=1)
td = env.reset()

my_obs = env.jax_obs["player_0"]
m1 = np.asarray(my_obs.units_mask)[0, 0]
m2 = np.asarray(my_obs.units_mask)[0, 1]
print(f"P0 Own units mask: {m1.sum()}, Enemy units mask: {m2.sum()}")

my_obs = env.jax_obs["player_1"]
m1 = np.asarray(my_obs.units_mask)[0, 0]
m2 = np.asarray(my_obs.units_mask)[0, 1]
print(f"P1 Own units mask: {m1.sum()}, Enemy units mask: {m2.sum()}")

