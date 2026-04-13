import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces
from torchrl.envs import PettingZooWrapper
import torch

class Dummy(ParallelEnv):
    def __init__(self):
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = self.possible_agents[:]
        self.observation_spaces = {
            a: spaces.Dict({
                "observation": spaces.Box(low=0, high=1, shape=(2,)),
                "action_mask": spaces.Box(low=0, high=1, shape=(3,), dtype=np.int8)
            }) for a in self.agents
        }
        self.action_spaces = {a: spaces.Discrete(3) for a in self.agents}
    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        obs = {a: {"observation": np.zeros(2, dtype=np.float32), "action_mask": np.ones(3, dtype=np.int8)} for a in self.agents}
        return obs, {a: {} for a in self.agents}
    def step(self, actions):
        obs = {a: {"observation": np.zeros(2, dtype=np.float32), "action_mask": np.ones(3, dtype=np.int8)} for a in self.agents}
        rews = {a: 0.0 for a in self.agents}
        terms = {a: False for a in self.agents}
        truncs = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, rews, terms, truncs, infos
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    def action_space(self, agent):
        return self.action_spaces[agent]

env = PettingZooWrapper(Dummy())
print(env.observation_spec)
print(env.action_spec)
td = env.reset()
print(td)
