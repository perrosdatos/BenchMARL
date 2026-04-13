import torch
from benchmarl.environments.lux.lux_env import LuxTorchRLEnv
env = LuxTorchRLEnv(batch_size=4, max_steps=150, match_count=5)
td = env.reset()
done_count = 0
for i in range(2000):
    a = torch.zeros((*env.action_spec.shape, 16), dtype=torch.long)
    td["action"] = a
    td["agents"] = {"action": a}
    td = env.step(td).get("next")
    done_count += td["done"].sum().item()
print("After 2000 steps per worker, total dones encountered:", done_count)
