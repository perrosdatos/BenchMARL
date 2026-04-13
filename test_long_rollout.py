import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from benchmarl.environments.lux.lux_env import LuxTorchRLEnv
from torchrl.envs.transforms import RewardSum, TransformedEnv, Compose
from torchrl.collectors import SyncDataCollector

print("Initializing TorchRL framework with integrated shaping modules...")
env = LuxTorchRLEnv(batch_size=4, max_steps=150, match_count=5)
env = TransformedEnv(env, Compose(RewardSum(in_keys=[("agents", "reward")], reset_keys=["_reset"])))

collector = SyncDataCollector(
    env,
    env.rand_action,
    frames_per_batch=1000,
    total_frames=10000,
)

print("Starting Collection loop...")
for i, batch in enumerate(collector):
    dones = batch.get(("next", "done"))
    rew = batch.get(("next", "agents", "reward"))
    
    # We will compute the absolute sum of instantaneous shaped rewards received
    total_reward_mag = torch.sum(torch.abs(rew))
    nonzero_count = torch.sum(rew != 0).item()
    
    if total_reward_mag > 0:
        print(f"Batch {i}: Collected reward absolute mass -> {total_reward_mag:.4f} (Non-zero signals: {nonzero_count})")
    
    if dones.any():
        term_r = batch.get(("next", "agents", "episode_reward"))
        print(f"  --> Batch {i} contains terminal states. Total done count: {dones.sum().item()}")
