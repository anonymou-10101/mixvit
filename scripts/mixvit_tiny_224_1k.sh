torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=23456 train.py /path/to/imagenet-1k \
         --model mixvit_t_224 --img-size 224 --epochs 300 --batch-size 128 \
         --opt adamw --weight-decay .05 --lr 3e-3 --sched cosine --warmup-epochs 30 \
         --warmup-lr 1e-6 --min-lr 1e-5 --drop-path .2 --smoothing 0.1 \
         --mixup .5 --cutmix 1.0 --clip-grad 1.0 \
         --aa rand-m9-mstd0.5-inc1 --reprob 0.25 --remode pixel \
         -tb 2048 --pin-mem --amp -j 8 --channels-last 
