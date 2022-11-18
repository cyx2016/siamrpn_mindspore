#!/bin/bash
export DEVICE_ID=$1
export DATA_NAME=$2
export MODEL_PATH=$3
export FILENAME=$4
python  eval.py  --device_id=$DEVICE_ID --dataset_path=$DATA_NAME --checkpoint_path=$MODEL_PATH --filename=$FILENAME &> eval.log &

