#!/bin/bash

echo "--- Exporting static environment variables... ---"

export VLLM_ATTENTION_BACKEND=XFORMERS

export WANDB_MODE=offline

echo "--- Starting evaluation loop for $TOTAL_RUNS runs... ---"

for i in $(seq 1 $TOTAL_RUNS)
do

    NEW_SEED=$(shuf -i 10000000-99999999 -n 1)

    export SEED=$NEW_SEED

    echo ""
    echo "================================================="
    echo "  Running Iteration $i / $TOTAL_RUNS with SEED=$SEED"
    echo "================================================="

    bash ./scripts/eval7b.sh

    echo "--- Iteration $i finished. ---"
done

echo ""
echo "--- All $TOTAL_RUNS iterations completed. ---"