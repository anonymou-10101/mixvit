#!/usr/bin/env bash

set -x

DATA_PATH=$1
GPUS=${2:-4}


if [ ${GPUS} -eq 2 ]; then
    GRAD_ACCUM_STEPS=8
elif [ ${GPUS} -eq 4 ]; then
    GRAD_ACCUM_STEPS=4
elif [ ${GPUS} -eq 8 ]; then
    GRAD_ACCUM_STEPS=2
else
    GRAD_ACCUM_STEPS=16
fi

torchrun --nproc_per_node=${GPUS} --master_addr=127.0.0.1 --master_port=23456 train.py ${DATA_PATH} \
         --model mixvit_b_224 --img-size 224 --epochs 300 --batch-size 128 \
         --opt adamw --weight-decay .05 --lr 3e-3 --sched cosine --warmup-epochs 30 \
         --warmup-lr 1e-6 --min-lr 1e-5 --drop-path .7 --smoothing 0.1 \
         --mixup .5 --cutmix 1.0 --clip-grad 1.0 \
         --aa rand-m9-mstd0.5-inc1 --reprob 0.25 --remode pixel \
         --grad-accum-steps ${GRAD_ACCUM_STEPS} --pin-mem --amp -j 8 --channels-last 
