export N_GPUS=8
export NNODES=1
export BASE_MODEL=${MODEL_PATH}
export TRAIN_DATA_PATH=./data/train.parquet
export TP_SIZE=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export RM_TYPE=default
export N_MDP=1
export BS=128
export EXPERIMENT_NAME=${DATA}-1.5b-${LENGTHM}-${TEMP}-${SEED}
export DATA_PATH=./data/${DATA}.parquet

bash ./scripts/validate_only_7b.sh