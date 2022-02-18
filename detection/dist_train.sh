#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#CUDA_VISIBLE_DEVICES=2,3,4 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=50020 \
#    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=50019 \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
