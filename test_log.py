import torch
from tensordict import TensorDict
from benchmarl.environments.lux.common import LuxClass

# Simulate batch
b_size = 2
done = torch.tensor([[False], [True]])
info_dict = {
    "agent_points": torch.tensor([[10.0], [50.0]]),
    "opponent_points": torch.tensor([[5.0], [40.0]]),
    "rc_fog_discovery": torch.tensor([[0.1], [0.5]])
}

batch = TensorDict({
    "next": TensorDict({
        "done": done,
        "info": TensorDict(info_dict, batch_size=[b_size])
    }, batch_size=[b_size])
}, batch_size=[b_size])

res = LuxClass.log_info(batch)
print("LOG RESULT:", res)
