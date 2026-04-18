#!/bin/bash
# Script de entrenamiento secuencial nocturno para Tesis (V2 - 12 Canales)
# Creado para aislar los procesos uno tras otro y liberar la VRAM limpia.

# echo "=========================================="
# echo "🚀 INICIANDO MASAC [1/3]..."
# echo "=========================================="
# XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_START_METHOD=thread python benchmarl/run.py \
#     algorithm=masac task=lux/match model=layers/cnn model@critic_model=layers/cnn \
#     experiment.sampling_device=cpu experiment.train_device=cuda experiment.buffer_device=cpu \
#     experiment.evaluation_episodes=2 experiment.off_policy_collected_frames_per_batch=1000 \
#     experiment.off_policy_n_envs_per_worker=4 experiment.off_policy_train_batch_size=128 \
#     experiment.off_policy_memory_size=7000 experiment.off_policy_n_optimizer_steps=45 \
#     experiment.wandb_extra_kwargs.name="lux_v2_masac_seq"

echo "=========================================="
echo "🚀 INICIANDO MAPPO [2/3]..."
echo "=========================================="
XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_START_METHOD=thread python benchmarl/run.py \
    algorithm=mappo task=lux/match model=layers/cnn model@critic_model=layers/cnn \
    experiment.max_n_frames=6000000 experiment.sampling_device=cpu experiment.train_device=cuda \
    experiment.buffer_device=cpu experiment.evaluation_episodes=2 experiment.on_policy_collected_frames_per_batch=1000 \
    experiment.on_policy_n_envs_per_worker=4 experiment.on_policy_minibatch_size=100 \
    experiment.wandb_extra_kwargs.name="lux_v2_mappo_seq"

echo "=========================================="
echo "🚀 INICIANDO QMIX [3/3]..."
echo "=========================================="
XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_START_METHOD=thread python benchmarl/run.py \
    algorithm=qmix task=lux/match model=layers/cnn model@critic_model=layers/cnn \
    experiment.max_n_frames=6000000 experiment.sampling_device=cpu experiment.train_device=cuda \
    experiment.buffer_device=cpu experiment.evaluation_episodes=2 experiment.off_policy_collected_frames_per_batch=1000 \
    experiment.off_policy_n_envs_per_worker=4 experiment.off_policy_train_batch_size=128 \
    experiment.off_policy_memory_size=7000 experiment.off_policy_n_optimizer_steps=45 \
    experiment.wandb_extra_kwargs.name="lux_v2_qmix_seq"

echo "=========================================="
echo "✅ ENTRENAMIENTO SECUENCIAL COMPLETADO"
echo "=========================================="
