#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export OUTPUT_PATH=output

for ds in scifact fever vc
do
    for shot in 90
    do
        for seed in 0 2 3 4
        do
            for st in 1500 
            do
                r=$((${st}/(${shot}/4)))
                python -u pl_train.py -k exp_name=${ds}_shots${shot}_seed${seed}_zs few_shot_random_seed=${seed} seed=${seed} dataset=${ds} batch_size=1 grad_accum_factor=2 num_steps=${st} eval_batch_size=4 num_shot=${shot} stage=2 zero_shot=true eval_epoch_interval=${r}
            done
        done
    done
done