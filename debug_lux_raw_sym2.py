import jax
import jax.tree_util
jax.tree_map = jax.tree_util.tree_map

from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import EnvParams
env = LuxAIS3Env(fixed_env_params=EnvParams(max_steps_in_match=15))
obs, state = env.reset(jax.random.PRNGKey(0))

import numpy as np

pts0 = obs["player_0"].team_points
pts1 = obs["player_1"].team_points
print(f"P0 points: {pts0}")
print(f"P1 points: {pts1}")

rn_mask0 = obs["player_0"].relic_nodes_mask
rn_mask1 = obs["player_1"].relic_nodes_mask
print(f"P0 relic_mask: {rn_mask0}")
print(f"P1 relic_mask: {rn_mask1}")

