#!/bin/bash
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh DATA_PATH DEVICE_ID"
echo "For example: bash run.sh data_path 0"
echo "It is better to use the absolute path."
echo "=============================================================================================================="

DATA_PATH=$1
DEVICE_ID=$2
export DEVICE_ID=$DEVICE_ID
python3 train.py --train_url=$DATA_PATH --device_id=$DEVICE_ID > train.log 2>&1 &
