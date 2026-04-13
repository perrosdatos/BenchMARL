import torch
from benchmarl.environments.lux.lux_env import LuxTorchRLEnv
from torchrl.envs.transforms import RewardSum, TransformedEnv, Compose
from torchrl.collectors import SyncDataCollector

env = LuxTorchRLEnv(batch_size=4, max_steps=15, match_count=1)
env = TransformedEnv(env, Compose(RewardSum(in_keys=[("agents", "reward")], reset_keys=["_reset"])))

collector = SyncDataCollector(
    env,
    env.rand_action,
    frames_per_batch=100,
    total_frames=300,
)
for i, batch in enumerate(collector):
    dones = batch.get(("next", "done"))
    if dones.any():
        print(f"Batch {i}: episode ended!")
        # check episode_reward
        r = batch.get(("next", "agents", "episode_reward"))
        print("Episode reward shape:", r.shape)
        print("Mean reward at done:", r.mean(-2).squeeze(-1)[dones.squeeze(-1)])
    else:
        print(f"Batch {i}: no episode ended")
