#!/bin/bash

if [ $# -ne 5 ]; then
    echo "Usage: $0 FIGHT_NAME \"BLACK_COMMAND\" \"WHITE_COMMAND\" NUM_GAMES GPU_ID"
    exit 1
fi

fight_name=$1
black_command=$2
white_command=$3
num_games=$4
gpu_id=$5

mkdir -p fight/$fight_name
echo "CUDA_VISIBLE_DEVICES=${gpu_id} gogui-twogtp -black \"$black_command\" -white \"$white_command\" -games $num_games -sgffile fight/$fight_name/$fight_name -alternate -auto -size 19 -komi 7.5 -threads 1 -verbose"
CUDA_VISIBLE_DEVICES=${gpu_id} gogui-twogtp -black "$black_command" -white "$white_command" -games $num_games -sgffile fight/$fight_name/$fight_name -alternate -auto -size 19 -komi 7.5 -threads 1 -verbose
