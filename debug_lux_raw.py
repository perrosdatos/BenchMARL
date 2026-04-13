import jax
import jax.tree_util
# monkey patch jax
jax.tree_map = jax.tree_util.tree_map

from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import EnvParams
try:
    env = LuxAIS3Env(fixed_env_params=EnvParams(max_steps_in_match=15))
    obs, state = env.reset(jax.random.PRNGKey(0))

    import numpy as np
    actions = {
        "player_0": jax.numpy.zeros((16, 3), dtype=jax.numpy.int32),
        "player_1": jax.numpy.zeros((16, 3), dtype=jax.numpy.int32)
    }

    for i in range(5):
        obs, state, r, t, trunc, info = env.step(jax.random.PRNGKey(i), state, actions)
        
        mask0 = obs["player_0"].units_mask[0]
        mask1 = obs["player_0"].units_mask[1]
        
        print(f"Step {i}:")
        print(f"  Team 0 mask sum: {mask0.sum()}")
        print(f"  Team 1 mask sum: {mask1.sum()}")
except Exception as e:
    print(e)
