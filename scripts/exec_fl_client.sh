#!/bin/bash
# python fl_client.py --gpu 0 --bsz 8 --epochs 5 --no-use-mask --ignore-pad --gradient-clip-val 1.0 --num-encoder-layers 1 --d-model 128 --lr 1e-3 --weight-decay 1e-6 --period 212 --language-model neuralmind/bert-base-portuguese-cased --text --no-visual --rnn-type lstm --shuffle --name SIL-LSTM-Instagram --wandb --dataset instagram --no-mil

if [ $# -eq 4 ]
    then
        echo "Experiment name: '$1'"
        echo "GPUs $2 $3 $4" 
    else
        echo "You must pass exactly 3 GPU indices"
        exit 1
fi

# if [ $# -eq 6 ]
#     then
#         echo "Experiment name: '$1'"
#         echo "GPUs $2 $3 $4 $5 $6" 
#     else
#         echo "You must pass exactly 5 GPU indices"
#         exit 1
# fi

source /home/paulomann/anaconda3/bin/activate demil
# GPUS=("$@")
GPUS=("${@:2}")

UUID=$(cat /proc/sys/kernel/random/uuid)

# for i in `seq 0 4`; do
for i in `seq 0 2`; do
    echo "Starting client $i in GPU ${GPUS[$i]}"
    # if [ $i -eq 2 ]
    #     then
    #         python fl_client.py --gpu ${GPUS[$i]} --bsz 32 --epochs 10 --no-use-mask --ignore-pad --gradient-clip-val 1.0 \
    #         --num-encoder-layers 1 --d-model 128 --lr 1e-3 --weight-decay 1e-6 --period 212 \
    #         --language-model neuralmind/bert-base-portuguese-cased --text --no-visual --rnn-type lstm \
    #         --shuffle --name $1-$UUID --wandb --dataset federated/instagram-stratified-MIL/$i --no-mil --seed 923 &
    #     else
    #         ( python fl_client.py --gpu ${GPUS[$i]} --bsz 32 --epochs 10 --no-use-mask --ignore-pad --gradient-clip-val 1.0 \
    #         --num-encoder-layers 1 --d-model 128 --lr 1e-3 --weight-decay 1e-6 --period 212 \
    #         --language-model neuralmind/bert-base-portuguese-cased --text --no-visual --rnn-type lstm \
    #         --shuffle --name $1-$UUID --wandb --dataset federated/instagram-stratified-MIL/$i --no-mil --seed 923 & ) > /dev/null 2>&1
    # fi

    python fl_client.py --gpu ${GPUS[$i]} --bsz 64 --epochs 5 --no-use-mask --ignore-pad --gradient-clip-val 1.0 \
    --num-encoder-layers 1 --d-model 128 --lr 3e-3 --weight-decay 1e-6 --period 212 \
    --language-model neuralmind/bert-base-portuguese-cased --text --no-visual --rnn-type lstm \
    --shuffle --name $1-$UUID --wandb --dataset federated/DepressBR-stratified-MIL/$i --no-mil --seed 738922 &
    # python fl_client.py --gpu ${GPUS[$i]} --bsz 32 --epochs 5 --no-use-mask --ignore-pad --gradient-clip-val 1.0 \
    # --num-encoder-layers 1 --d-model 128 --lr 2e-5 --weight-decay 1e-6 --period 212 \
    # --language-model neuralmind/bert-base-portuguese-cased --text --no-visual --rnn-type lstm \
    # --shuffle --name $1-$UUID --wandb --dataset federated/instagram-stratified-MIL/$i --no-mil --seed 491851 &
done