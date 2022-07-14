#!/bin/bash
# python fl_client.py --gpu 0 --bsz 8 --epochs 5 --no-use-mask --ignore-pad --gradient-clip-val 1.0 --num-encoder-layers 1 --d-model 128 --lr 1e-3 --weight-decay 1e-6 --period 212 --language-model neuralmind/bert-base-portuguese-cased --text --no-visual --rnn-type lstm --shuffle --name SIL-LSTM-Instagram --wandb --dataset instagram --no-mil

# if [ $# -eq 5 ]
#     then
#         echo "GPUs $1 $2 $3 $4 $5" 
#     else
#         echo "You must pass exactly 5 GPU indices"
#         exit 1
# fi

source /home/paulomann/anaconda3/bin/activate demil
GPUS=("$@")

UUID=$(cat /proc/sys/kernel/random/uuid)

# for i in `seq 0 4`; do
for i in `seq 0 1`; do
    echo "Starting client $i in GPU ${GPUS[$i]}"
    python fl_client.py --gpu ${GPUS[$i]} --bsz 32 --epochs 1 --no-use-mask --ignore-pad --gradient-clip-val 1.0 \
    --num-encoder-layers 1 --d-model 128 --lr 1e-3 --weight-decay 1e-6 --period 212 \
    --language-model neuralmind/bert-base-portuguese-cased --text --no-visual --rnn-type lstm \
    --shuffle --name SIL-LSTM-Instagram-$UUID --wandb --dataset federated/instagram-random-SIL/$i --no-mil &
done