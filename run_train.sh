#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex
export NCCL_SOCKET_TIMEOUT=-1
export NCCL_COMM_TIMEOUT=-1
export NCCL_DEBUG=warn
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_HCA=mlx5
# export NCCL_CROSS_NIC=1
# export NCCL_ALGO=Tree
export NCCL_ALGO="tree;allgather:ring;broadcast:ring;reducescatter:ring"

if [ "${MULTI_NODE:-0}" = "1" ]; then
    export MY_NODE_RANK=${RANK}
    export TOTAL_NODES=${WORLD_SIZE}
    MASTER_ADDR=${MASTER_ADDR}
    MASTER_PORT=${MASTER_PORT}
    NNODES=${WORLD_SIZE}
    NODE_RANK=${RANK}
else
    export MY_NODE_RANK=0
    export TOTAL_NODES=1
    MASTER_ADDR=localhost
    MASTER_PORT=6001
    NNODES=1
    NODE_RANK=0
fi

export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export LOG_RANK=${LOG_RANK:-0}
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NNODES=${NNODES}"
echo "NODE_RANK=${NODE_RANK}"

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT --rdzv_id=pytorchddp --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} --local-ranks-filter ${LOG_RANK} --role rank"

cat /2214/wandb.netrc > /root/.netrc
export WANDB_MODE=offline

export TORCH_HOME='/2214/torch'
export HF_HOME='/2214/huggingface'

# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_train.sh
# NGPU=${NGPU:-"6"}

CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama3/train_configs/music_bark_2dot5b.toml"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

# TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd /2214/dongyuanliang/torchtitan
# TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
/2214/conda_envs/torchtitan/bin/torchrun $DISTRIBUTED_ARGS \
-m torchtitan.train --job.config_file ${CONFIG_FILE} $overrides 2>&1 | tee -a run_${NODE_RANK}.log_music_bark_2dot5B_fixdatasampler
# --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
tail -f /dev/null