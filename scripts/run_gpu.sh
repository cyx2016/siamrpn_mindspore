#!/bin/bash

if [ $# != 1 ]
then 
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash run.sh DEVICE_ID"
    echo "For example: bash run_gpu.sh 0"
    echo "=============================================================================================================="
exit 1
fi


DEVICE_ID=$1

export DEVICE_ID=$DEVICE_ID
python3 train.py --device_id=$DEVICE_ID --device_target="GPU"> train.log 2>&1 &
