#!/bin/bash
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh DATA_PATH RANK_TABLE"
echo "For example: bash run.sh /path/dataset /path/rank_table"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
DATA_PATH=$1
export RANK_SIZE=8
RANK_TABLE=$(get_real_path $2)

EXEC_PATH=$(pwd)
echo "$EXEC_PATH"
export RANK_TABLE_FILE=$RANK_TABLE

start_divice=0
for((i=$start_divice;i<$[$start_divice+$RANK_SIZE];i++))
do
    rm -rf device$i
    mkdir device$i
    mkdir device$i/src
    cp ./train.py  ./device$i
    cp ./src/net.py ./src/loss.py ./src/config.py ./src/util.py ./src/data_loader.py ./src/generate_anchors.py ./device$i/src
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    python3 train.py  --train_url $DATA_PATH --is_parallel=True &> log &
    cd ../
done
echo "finish"
cd ../
