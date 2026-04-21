from benchmarl.experiment import Experiment
from benchmarl.conf.experiment import get_experiment_config
from benchmarl.environments.lux.lux_env import LuxTask
from benchmarl.models.cnn import CnnConfig
from benchmarl.algorithms.mappo import MappoConfig
import torch

run_dir = "outputs/2026-04-21/20-31-15/mappo_match_v2_cnn__3af9bf08_26_04_21-20_31_15"
import glob, os
checkpoints = glob.glob(os.path.join(run_dir, "checkpoints", "checkpoint_*.pt"))
checkpoints.sort(key=lambda x: int(os.path.basename(x).replace("checkpoint_", "").replace(".pt", "")))
latest_ckpt = checkpoints[-1]

exp_config = get_experiment_config("mappo_match_v2_cnn")
exp_config.evaluation = True
exp_config.restore_file = latest_ckpt

experiment = Experiment(
    task=LuxTask.MATCH_V2,
    algorithm_config=MappoConfig.get_from_yaml(),
    model_config=CnnConfig.get_from_yaml(),
    config=exp_config
)

td = experiment.test_env.rollout(max_steps=1, policy=None, break_when_any_done=False)
raw_obs = td.get(("agents", "observation")).shape
print("RAW OBS SHAPE:", raw_obs)

act = experiment.policy(td[..., -1])
print("ACT SHAPE IN TD:", act.shape)
