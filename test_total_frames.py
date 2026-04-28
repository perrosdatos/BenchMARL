import torch
from torchrl.collectors import SyncDataCollector
from torchrl.envs import SerialEnv, EnvBase
from tensordict.nn import TensorDictModule
import torch.nn as nn
from torchrl.envs.libs.gym import GymEnv
import sys

# Create a dummy env and policy
env = GymEnv("CartPole-v1")
policy = TensorDictModule(nn.Linear(4, 2), in_keys=["observation"], out_keys=["action"])

# Test 1: total_frames = 3M
try:
    c1 = SyncDataCollector(env, policy, frames_per_batch=1000, total_frames=3000000, device="cpu")
    print("3M total_frames OK")
except Exception as e:
    print("3M failed:", e)

# Test 2: total_frames = 6M
try:
    c2 = SyncDataCollector(env, policy, frames_per_batch=1000, total_frames=6000000, device="cpu")
    print("6M total_frames OK")
except Exception as e:
    print("6M failed:", e)

# Test 3: total_frames = 100M
try:
    c3 = SyncDataCollector(env, policy, frames_per_batch=1000, total_frames=100000000, device="cpu")
    print("100M total_frames OK")
except Exception as e:
    print("100M failed:", e)
