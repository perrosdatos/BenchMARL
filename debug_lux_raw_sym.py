import jax
import jax.tree_util
jax.tree_map = jax.tree_util.tree_map

from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import EnvParams
env = LuxAIS3Env(fixed_env_params=EnvParams(max_steps_in_match=15))
obs, state = env.reset(jax.random.PRNGKey(0))

import numpy as np
actions = {
    "player_0": jax.numpy.zeros((16, 3), dtype=jax.numpy.int32),
    "player_1": jax.numpy.zeros((16, 3), dtype=jax.numpy.int32)
}

for i in range(2):
    obs, state, _, _, _, _ = env.step(jax.random.PRNGKey(i), state, actions)
    
mask0_p0 = obs["player_0"].units_mask[0]
mask1_p0 = obs["player_0"].units_mask[1]

mask0_p1 = obs["player_1"].units_mask[0]
mask1_p1 = obs["player_1"].units_mask[1]

print(f"Player_0 Observation -> index 0 sum: {mask0_p0.sum()}, index 1 sum: {mask1_p0.sum()}")
print(f"Player_1 Observation -> index 0 sum: {mask0_p1.sum()}, index 1 sum: {mask1_p1.sum()}")
