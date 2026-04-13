import jax
from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import EnvParams
env = LuxAIS3Env(fixed_env_params=EnvParams(max_steps_in_match=5))
obs, state = env.reset(jax.random.PRNGKey(0))
for i in range(10):
    a = env.action_space().sample(jax.random.PRNGKey(i))
    obs, state, r, t, trunc, info = env.step(jax.random.PRNGKey(i), state, a)
    print(i, "steps:", getattr(state, "env_steps", getattr(state, "steps", None)))
    if t["player_0"] or trunc["player_0"]:
        print("Done evaluated TRUE!")
