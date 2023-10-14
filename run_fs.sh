#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export OUTPUT_PATH=output

for ds in fever scifact vc
do
    for seed in 0 2 3 4
    do
    python -u pl_train.py -k  exp_name=${ds}_shots${shot}_seed${seed}_fs few_shot_random_seed=${seed} seed=${seed} dataset=${ds} batch_size=1 grad_accum_factor=2 num_steps=1500 eval_batch_size=4 num_shot=12 stage=2 load_weight=pretrained_checkpoints/warmup/${ds}_shots${shot}_seed${seed}.pt
    python -u pl_train.py -k  exp_name=${ds}_shots${shot}_seed${seed}_fs few_shot_random_seed=${seed} seed=${seed} dataset=${ds}  save_model=false num_steps=0 eval_before_training=True eval_batch_size=4 num_shot=12 stage=2 load_weight=${OUTPUT_PATH}/${ds}_shots${shot}_seed${seed}_fs/finish.pt
    done
done
