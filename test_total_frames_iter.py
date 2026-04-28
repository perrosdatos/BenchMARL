import torch
from torchrl.collectors import SyncDataCollector
from torchrl.envs.libs.gym import GymEnv
from tensordict.nn import TensorDictModule
import torch.nn as nn
import os
import psutil

env = GymEnv("CartPole-v1")
policy = TensorDictModule(nn.Linear(4, 2), in_keys=["observation"], out_keys=["action"])

def test_collector(total_frames):
    c = SyncDataCollector(env, policy, frames_per_batch=1000, total_frames=total_frames, device="cpu")
    process = psutil.Process(os.getpid())
    for i, batch in enumerate(c):
        if i == 10: break
    print(f"Total Frames: {total_frames}, RSS: {process.memory_info().rss / 1e6} MB")

test_collector(3000000)
test_collector(6000000)
test_collector(100000000)
