#!/bin/bash

python -u train.py \
--batch-size 32 --dropout 0.1 --encoder-dropout 0.1 \
--pooler-dropout 0.1 --init-face --use-bn --epochs 100 \
--num-layers 3 --lr 0.001 --weight-decay 0.01  --beta2 0.999 \
--mlp-hidden-size 128 --lr-warmup --use-adamw --node-attn --gradmultiply -1 \
--edge-rep nefu \
--seed 42 \
--save-ckt \
--log-interval 10 \
--checkpoint-dir /tmp/ckpt \
--raw-data-path ../data/uspto50k \
