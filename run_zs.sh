#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export OUTPUT_PATH=output

for ds in scifact fever vc
do
    for seed in 0 2 3 4
    do
    r=$((${st}/(${shot}/4)))
    python -u pl_train.py -k exp_name=${ds}_shots90_seed${seed}_zs few_shot_random_seed=${seed} seed=${seed} dataset=${ds} batch_size=1 grad_accum_factor=2 num_steps=1500 eval_batch_size=4 num_shot=90 stage=2 zero_shot=true eval_epoch_interval=${r}
    done
done
