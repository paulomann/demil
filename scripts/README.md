
# To run a simple training 

Here, we do not ignore padding, and we mask the ith element to not see the (ith + x) element

`python run_training.py --gpu 0 --num-encoder-layers 1 --num-decoder-layers 1 --txt-freeze-n-layers 10 --vis-freeze-n-layers 8 --language-model xlm-roberta-base --use-mask --no-ignore-pad`