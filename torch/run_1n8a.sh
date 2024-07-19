#!/bin/bash
export OMP_NUM_THREADS=4
export MUSA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export MUSA_KERNEL_TIMEOUT=5400000
export NCCL_PROTOS=2
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH=$PYTHONPATH:/home/dist/FlagScale

torchrun \
    --nproc_per_node 8 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 12361 \
    test_collective.py
