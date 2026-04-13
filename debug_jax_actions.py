import jax
import jax.numpy as jnp
import numpy as np
b_size = 3
zeros_sap = np.zeros((b_size, 16, 2), dtype=np.int32)
actions = np.ones((b_size, 16), dtype=np.int32) * 4 # random action
action_3d = np.concatenate([actions[..., None], zeros_sap], axis=-1)

jax_actions_0 = np.zeros_like(action_3d)
jax_actions_1 = np.zeros_like(action_3d)

t_ids_np = np.array([0, 1, 0])

for b in range(b_size):
    if t_ids_np[b] == 0:
        jax_actions_0[b] = action_3d[b]
    else:
        jax_actions_1[b] = action_3d[b]

print("Actions 0:", jax_actions_0[:, 0, 0])
print("Actions 1:", jax_actions_1[:, 0, 0])
