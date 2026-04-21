#!/bin/bash
# Sequential night training script for Thesis (V2 - 16 Channels)
# Created to isolate processes sequentially and free up VRAM cleanly.
# Adjusted order: MAPPO -> QMIX -> MASAC

echo "=========================================="
echo "🚀 STARTING MAPPO [1/3]..."
echo "=========================================="
XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_START_METHOD=thread python benchmarl/run.py \
    algorithm=mappo task=lux/match_v2 model=layers/cnn_lux_16ch model@critic_model=layers/cnn_lux_16ch \
    experiment.max_n_frames=3000000 \
    experiment.sampling_device=cpu experiment.train_device=cuda \
    experiment.buffer_device=cpu experiment.evaluation_episodes=3 \
    experiment.on_policy_collected_frames_per_batch=1000 \
    experiment.on_policy_n_envs_per_worker=4 experiment.on_policy_minibatch_size=100 \
    experiment.checkpoint_interval=150000 experiment.keep_checkpoints_num=20 experiment.exclude_buffer_from_checkpoint=true experiment.project_name="Lux_Thesis_16CH_v2_BiggerCNN" experiment.wandb_extra_kwargs.name="mappo_16ch_biggercnn_3M_run"

echo "=========================================="
echo "🚀 STARTING QMIX [2/3]..."
echo "=========================================="
XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_START_METHOD=thread python benchmarl/run.py \
    algorithm=qmix task=lux/match_v2 model=layers/cnn_lux_16ch model@critic_model=layers/cnn_lux_16ch \
    experiment.max_n_frames=3000000 \
    experiment.sampling_device=cpu experiment.train_device=cuda \
    experiment.buffer_device=cpu experiment.evaluation_episodes=3 \
    experiment.off_policy_collected_frames_per_batch=400 \
    experiment.off_policy_n_envs_per_worker=4 experiment.off_policy_train_batch_size=128 \
    experiment.off_policy_memory_size=5000 experiment.off_policy_n_optimizer_steps=20 \
    experiment.checkpoint_interval=150000 experiment.keep_checkpoints_num=20 experiment.exclude_buffer_from_checkpoint=true experiment.project_name="Lux_Thesis_16CH_v2_BiggerCNN" experiment.wandb_extra_kwargs.name="qmix_16ch_biggercnn_3M_run"

echo "=========================================="
echo "🚀 STARTING MASAC [3/3]..."
echo "=========================================="
XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_START_METHOD=thread python benchmarl/run.py \
    algorithm=masac task=lux/match_v2 model=layers/cnn_lux_16ch model@critic_model=layers/cnn_lux_16ch \
    experiment.max_n_frames=3000000 \
    experiment.sampling_device=cpu experiment.train_device=cuda experiment.buffer_device=cpu \
    experiment.evaluation_episodes=3 experiment.off_policy_collected_frames_per_batch=1000 \
    experiment.off_policy_n_envs_per_worker=4 experiment.off_policy_train_batch_size=128 \
    experiment.off_policy_memory_size=5000 experiment.off_policy_n_optimizer_steps=45 \
    experiment.checkpoint_interval=150000 experiment.keep_checkpoints_num=20 experiment.exclude_buffer_from_checkpoint=true experiment.project_name="Lux_Thesis_16CH_v2_BiggerCNN" experiment.wandb_extra_kwargs.name="masac_16ch_biggercnn_3M_run"

echo "=========================================="
echo "✅ SEQUENTIAL TRAINING COMPLETED"
echo "=========================================="
