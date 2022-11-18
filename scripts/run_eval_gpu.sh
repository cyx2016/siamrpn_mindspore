#!/bin/bash

if [ $# != 4 ]
then 
    echo "Usage: bash run_eval_gpu.sh [DEVICE_id] [DATA_NAME] [MODEL_PATH] [FILENAME]"
exit 1
fi

export DEVICE_ID=$1
export DATA_NAME=$2
export MODEL_PATH=$3
export FILENAME=$4
python  eval.py  --device_id=$DEVICE_ID --dataset_path=$DATA_NAME --checkpoint_path=$MODEL_PATH --filename=$FILENAME --device_target="GPU" &> eval_gpu.log &

