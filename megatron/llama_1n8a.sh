#!/bin/bash 
export OMP_NUM_THREADS=4
export MUSA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export MUSA_KERNEL_TIMEOUT=5400000
export NCCL_PROTOS=2
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH=$PYTHONPATH:/home/dist/FlagScale

WORLD_SIZE=8

TP=${TP:-1}
PP=${PP:-1}
DP=$(($WORLD_SIZE/$TP/$PP))
MBS=${MBS:-1}
GBS=${GBS:-$(($WORLD_SIZE/$TP/$PP*$MBS))}
OPT=${OPT:-none}
MODEL_SIZE=${MODEL_SIZE:-7}
SEQLEN=${SEQLEN:-4096}

if [ "$MODEL_SIZE" == "1" ]; then
    HIDDEN_SIZE=4096
    FFN_HIDDEN_SIZE=11008
    LAYER=3
    NUM_HEAD=32
elif [ "$MODEL_SIZE" == "7" ]; then
    HIDDEN_SIZE=4096
    FFN_HIDDEN_SIZE=11008
    LAYER=32
    NUM_HEAD=64
elif [ "$MODEL_SIZE" == "13" ]; then
    HIDDEN_SIZE=5120
    FFN_HIDDEN_SIZE=13696
    LAYER=${LAYER:-40}
    NUM_HEAD=80
elif [ "$MODEL_SIZE" == "70" ]; then
    HIDDEN_SIZE=8192
    FFN_HIDDEN_SIZE=28672
    LAYER=80
    NUM_HEAD=128
else
    echo "Error: Invalid MODEL_SIZE value. Must be 7, 13, or 70."
    exit 1
fi

torchrun \
    --nproc_per_node $WORLD_SIZE \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 172.23.9.11 \
    --master_port 12361 \
    /home/dist/FlagScale/megatron/pretrain_gpt.py \
        --train-iters 3000 \
        --eval-iters 0 \
        --pipeline-model-parallel-size $PP \
        --tensor-model-parallel-size $TP \
        --num-layers $LAYER \
        --hidden-size $HIDDEN_SIZE \
        --ffn-hidden-size $FFN_HIDDEN_SIZE \
        --num-attention-heads $NUM_HEAD \
        --micro-batch-size $MBS \
        --global-batch-size $GBS \
        --seq-length $SEQLEN \
        --max-position-embeddings $SEQLEN \
        --disable-bias-linear \
        --use-distributed-optimizer \
        --distributed-backend mccl \
        --use-flash-attn \
        --sequence-parallel \
        --device-type mthreads \
        --no-gradient-accumulation-fusion \
        --bf16 \
        --attention-softmax-in-fp32 \
        --no-masked-softmax-fusion \
        --rotary-position-embeddings-in-fp32 \
        --data-path /home/dist/dataset/pile/pile_wikipedia_demo \
        --tokenizer-type AquilaTokenizer \
        --vocab-file ../aquila/tokenizer/vocab.json \
        --vocab-size 100008 \
        --merge-file ../aquila/tokenizer/merges.txt \
        --special-tokens-file ../aquila/tokenizer/special_tokens.txt \
        --data-impl mmap \
        --split 1 \
        --layernorm-epsilon 1e-5 \
        --use-rotary-position-embeddings \
        --no-position-embedding \
        --swiglu \
        --multiple-of 256 \
        --normalization RMSNorm \
        --apply-layernorm-rms \
        --untie-embeddings-and-output-weights \
        --init-method-std 0.02 \
        --seed 42 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --clip-grad 1.0 \
        --lr 3.5e-4 \
        --lr-decay-style cosine \
        --lr-warmup-fraction 0.01 \
        --min-lr 3.5e-5 \
        --loss-scale 1 \
        --save-interval 10000000 \
        --log-interval 1 \
        --log-params-norm

