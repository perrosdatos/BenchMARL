#!/bin/bash
# Script de entrenamiento secuencial nocturno para Tesis (V2 - 16 Canales)
# Creado para aislar los procesos uno tras otro y liberar la VRAM limpia.
# Orden ajustado: QMIX -> MAPPO -> MASAC

echo "=========================================="
echo "🚀 INICIANDO QMIX [1/3]..."
echo "=========================================="
XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_START_METHOD=thread python benchmarl/run.py \
    algorithm=qmix task=lux/match_v2 model=layers/cnn model@critic_model=layers/cnn \
    experiment.max_n_frames=3000000 \
    experiment.sampling_device=cpu experiment.train_device=cuda \
    experiment.buffer_device=cpu experiment.evaluation_episodes=3 \
    experiment.off_policy_collected_frames_per_batch=400 \
    experiment.off_policy_n_envs_per_worker=4 experiment.off_policy_train_batch_size=128 \
    experiment.off_policy_memory_size=5000 experiment.off_policy_n_optimizer_steps=20 \
    experiment.checkpoint_interval=150000 experiment.project_name="Lux_Thesis_16CH_v2" experiment.wandb_extra_kwargs.name="qmix_16ch_3M_run"

echo "=========================================="
echo "🚀 INICIANDO MAPPO [2/3]..."
echo "=========================================="
XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_START_METHOD=thread python benchmarl/run.py \
    algorithm=mappo task=lux/match_v2 model=layers/cnn model@critic_model=layers/cnn \
    experiment.max_n_frames=3000000 \
    experiment.sampling_device=cpu experiment.train_device=cuda \
    experiment.buffer_device=cpu experiment.evaluation_episodes=3 \
    experiment.on_policy_collected_frames_per_batch=1000 \
    experiment.on_policy_n_envs_per_worker=4 experiment.on_policy_minibatch_size=100 \
    experiment.checkpoint_interval=150000 experiment.project_name="Lux_Thesis_16CH_v2" experiment.wandb_extra_kwargs.name="mappo_16ch_3M_run"

echo "=========================================="
echo "🚀 INICIANDO MASAC [3/3]..."
echo "=========================================="
XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_START_METHOD=thread python benchmarl/run.py \
    algorithm=masac task=lux/match_v2 model=layers/cnn model@critic_model=layers/cnn \
    experiment.max_n_frames=3000000 \
    experiment.sampling_device=cpu experiment.train_device=cuda experiment.buffer_device=cpu \
    experiment.evaluation_episodes=3 experiment.off_policy_collected_frames_per_batch=1000 \
    experiment.off_policy_n_envs_per_worker=4 experiment.off_policy_train_batch_size=128 \
    experiment.off_policy_memory_size=5000 experiment.off_policy_n_optimizer_steps=45 \
    experiment.checkpoint_interval=150000 experiment.project_name="Lux_Thesis_16CH_v2" experiment.wandb_extra_kwargs.name="masac_16ch_3M_run"

echo "=========================================="
echo "✅ ENTRENAMIENTO SECUENCIAL COMPLETADO"
echo "=========================================="
