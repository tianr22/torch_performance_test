#!/bin/bash
set -x
srun $SRUN_ARGS \
    --exclusive=user -N ${N:-1} --ntasks-per-node=8 --gres=gpu:8 \
    ./wrapper.sh python3 -u test_collective.py
