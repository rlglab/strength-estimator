#!/bin/bash
set -e

usage()
{
	echo "Usage: $0 GAME_TYPE CONFIGURE_FILE [OPTION]..."
	echo "The zero-server manages the training session and organizes connected zero-workers."
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

# create default name of training if name is not assigned
executable_file=build/${game_type}/strength_${game_type}
if [[ -z ${train_dir} ]]; then
	train_dir=${name_prefix}$(${executable_file} -mode zero_training_name -conf_file ${configure_file} -conf_str "${overwrite_conf_str}" 2>/dev/null)${name_suffix}
fi

run_stage="R"
if [ -d ${train_dir} ]; then
	read -n1 -p "${train_dir} has existed. (R)estart / (C)ontinue / (Q)uit? " run_stage
	echo ""
fi

model_file="weight_iter_0.pkl"
if [[ ${run_stage,} == "r" ]]; then
	rm -rf ${train_dir}

	echo "create ${train_dir} ..."
	mkdir -p ${train_dir}/model
	touch ${train_dir}/op.log
	new_configure_file=$(basename ${train_dir}).cfg
	${executable_file} -gen ${train_dir}/${new_configure_file} -conf_file ${configure_file} -conf_str "${overwrite_conf_str}" 2>/dev/null

	# setup initial weight
	PYTHONPATH=. python strength/trainer/train.py ${game_type} ${train_dir} "" ${train_dir}/${new_configure_file} 2>&1
elif [[ ${run_stage,} == "c" ]]; then
	model_file=$(ls ${train_dir}/model/ | grep ".pkl$" | sort -V | tail -n1)
	new_configure_file=$(basename ${train_dir}).cfg
	echo y | ${executable_file} -gen ${train_dir}/${new_configure_file} -conf_file ${train_dir}/${new_configure_file} -conf_str "${overwrite_conf_str}" 2>/dev/null

	# friendly notification if continuing training
	read -n1 -p "Continue training from model file: ${model_file}, configuration: ${train_dir}/${new_configure_file}. Sure? (y/n) " yn
	[[ ${yn,,} == "y" ]] || exit
	echo ""
else
	exit
fi

# run training
cuda_devices=$(echo ${gpu_list} | awk '{ split($0, chars, ""); printf(chars[1]); for(i=2; i<=length(chars); ++i) { printf(","chars[i]); } }')
echo "CUDA_VISIBLE_DEVICES=${cuda_devices} PYTHONPATH=. python strength/trainer/train.py ${game_type} ${train_dir} ${train_dir}/${new_configure_file}"
CUDA_VISIBLE_DEVICES=${cuda_devices} PYTHONPATH=. python strength/trainer/train.py ${game_type} ${train_dir} ${model_file} ${train_dir}/${new_configure_file} 2>&1 | tee -a ${train_dir}/op.log
