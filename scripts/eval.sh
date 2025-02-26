#!/bin/bash
set -e

usage()
{
	echo "Usage: $0 GAME_TYPE CONFIGURE_FILE [OPTION]..."
	echo "The eval do the testing and plot the results."
	echo ""
	echo "Required arguments:"
	echo "  GAME_TYPE: $(find ./ ../ -maxdepth 2 -name build.sh -exec grep -m1 support_games {} \; -quit | sed -E 's/.+\("|"\).*//g;s/" "/, /g')"
	echo "  CONFIGURE_FILE: the configure file (*.cfg) to use"
	echo ""
	echo "Optional arguments:"
	echo "  -h,        --help                 Give this help list"
    echo "  -g,        --gpu                  Assign available GPUs for worker, e.g. 0123"
	echo "  -n,        --name                 Assign name for training directory"
	echo "  -np,       --name_prefix          Add prefix name for default training directory name"
	echo "  -ns,       --name_suffix          Add suffix name for default training directory name"
    echo "  -conf_str                         Overwrite settings in the configure file"
	exit 1
}

# check argument
if [ $# -lt 2 ] || [ $(($# % 2)) -eq 1 ]; then
	usage
else
	game_type=$1; shift
	configure_file=$1; shift

    # default arguments
    num_gpu=$(nvidia-smi -L | wc -l)
    gpu_list=$(echo $num_gpu | awk '{for(i=0;i<$1;i++)printf i}')
fi

train_dir=""
name_prefix=""
name_suffix=""
overwrite_conf_str=""
while :; do
	case $1 in
		-h|--help) shift; usage
		;;
		-g|--gpu) shift; gpu_list=$1; num_gpu=${#gpu_list}
		;;
		-n|--name) shift; train_dir=$1
		;;
		-np|--name_prefix) shift; name_prefix=$1
		;;
		-ns|--name_suffix) shift; name_suffix=$1
		;;
		-conf_str) shift; overwrite_conf_str=$1
		;;
		"") break
		;;
		*) echo "Unknown argument: $1"; usage
		;;
	esac
	shift
done

# setup
[ -d eval ] || mkdir eval
executable_file=build/${game_type}/strength_${game_type}
output_file_prefix=$(echo "quit" | ${executable_file} -conf_file ${configure_file} -conf_str "${overwrite_conf_str}" 2>&1 | grep "^nn_file_name" | awk -F "#|=|\." '{ print $2 }' | sed 's/\//_/g')

if [ ! -f eval/${output_file_prefix}.log ]; then
    # run evaluator
    cuda_devices=$(echo ${gpu_list} | awk '{ split($0, chars, ""); printf(chars[1]); for(i=2; i<=length(chars); ++i) { printf(","chars[i]); } }')
    echo "CUDA_VISIBLE_DEVICES=${cuda_devices} ${executable_file} -conf_file ${configure_file} -conf_str "${overwrite_conf_str}" -mode evaluator | tee eval/${output_file_prefix}.log"
    CUDA_VISIBLE_DEVICES=${cuda_devices} ${executable_file} -conf_file ${configure_file} -conf_str "${overwrite_conf_str}" -mode evaluator | tee eval/${output_file_prefix}.log

    # draw plot
    echo "./scripts/plot.py eval/${output_file_prefix}.log eval/${output_file_prefix}.png"
    ./scripts/plot.py eval/${output_file_prefix}.log eval/${output_file_prefix}.png
else
    echo "eval/${output_file_prefix}.log already exists, skip"
    exit 0
fi
