# demil

## Initialization:
 - Run this command line:
 
 ```
 python run_training.py --gpu 5 --bsz 64 --epochs 50 --no-use-mask --ignore-pad --gradient-clip-val 1 --nhead 2 --num-encoder-layers 4 --d-model 512 --lr 5e-4 --weight-decay 1e-5 --period -1 --language-model bert-base-cased --no-mil --rnn-type bert --shuffle --name eRisk2021 --no-wandb --text --no-visual --pos-embedding relative_key_query --dataset eRisk2021

  python run_training.py --gpu 5 --bsz 64 --epochs 50 --no-use-mask --ignore-pad --gradient-clip-val 1 --nhead 2 --num-encoder-layers 4 --d-model 512 --lr 5e-4 --weight-decay 1e-5 --period -1 --language-model bert-base-cased --no-mil --rnn-type bert --shuffle --name eRisk2021 --no-wandb --text --no-visual --pos-embedding relative_key_query --dataset LOSADA2016

  python fl_client.py --gpu 6 --bsz 64 --epochs 5 --no-use-mask --ignore-pad --gradient-clip-val 1 --nhead 2 --num-encoder-layers 4 --d-model 512 --lr 5e-4 --weight-decay 1e-5 --period -1 --language-model bert-base-cased --no-mil --rnn-type bert --shuffle --name eRisk2021 --no-wandb --text --no-visual --pos-embedding relative_key_query --dataset eRisk2021

  python fl_client.py --gpu 7 --bsz 64 --epochs 5 --no-use-mask --ignore-pad --gradient-clip-val 1 --nhead 2 --num-encoder-layers 4 --d-model 512 --lr 5e-4 --weight-decay 1e-5 --period -1 --language-model bert-base-cased --no-mil --rnn-type bert --shuffle --name LOSADA2016 --no-wandb --text --no-visual --pos-embedding relative_key_query --dataset LOSADA2016

  python run_training.py --gpu 7 --bsz 64 --epochs 5 --no-use-mask --ignore-pad --gradient-clip-val 1 --nhead 2 --num-encoder-layers 4 --d-model 512 --lr 5e-4 --weight-decay 1e-5 --period -1 --language-model fasttext --no-mil --rnn-type bert --shuffle --name eRisk2021 --no-wandb --text --no-visual --pos-embedding relative_key_query --dataset eRisk2021
 ```

## TODO : 

 - O log do servidor diz centralizar os scores, mas não mostra os scores centralizados. Talvez isso seja um print comentado, senão, eu posso botar um print no final do processo para mostrar esses scores. (Não é necessário, mas Rafaela me respondeu como fazer isso)

 - Trocar do BERT para FastText, com embeddings do BERT

 - Ler estatisticas do dataset eRisk2021 + LOSADA2016 (Distribuição de tokens*, distribuição de classes, distribuição de datas)

 - - Tokenização, do jeito que você entendeu mesmo (usar SPACY) **Priorizar**

 - - Passar pelo bert tokenizer para saber como precisamos cortar os tokens (usar BERT)

 - Mandar as estatísticas por e-mail

 - Rodar Centralizado

 - Downsampling

 - Federado




## Error:

Corrected (added sanity check to model.py line 598)

## Run Metrics:

### First Run (50 epochs, using eRisk2021)
#===================================================#
       Test metric             DataLoader 0
#===================================================#
         fscore             0.15384615384615385
        precision                   1.0
         recall             0.08333333333333333
        test_loss           0.7123931050300598
#===================================================#

### Second Run (50 epochs, using LOSADA2016)
#===================================================#
       Test metric             DataLoader 0
#===================================================#
         fscore             0.23478260869565218
        precision           0.1330049261083744
         recall                     1.0
        test_loss           0.26314640045166016
#===================================================#

### Third Run (Server Output)

`
INFO flower 2023-01-13 20:37:57,207 | app.py:134 | Flower server running (2 rounds), SSL is disabled
INFO flower 2023-01-13 20:37:57,207 | server.py:84 | Initializing global parameters
INFO flower 2023-01-13 20:37:57,207 | server.py:256 | Requesting initial parameters from one random client
INFO flower 2023-01-13 20:38:50,832 | server.py:259 | Received initial parameters from one random client
INFO flower 2023-01-13 20:38:50,833 | server.py:86 | Evaluating initial parameters
INFO flower 2023-01-13 20:38:50,833 | server.py:99 | FL starting
DEBUG flower 2023-01-13 20:41:21,043 | server.py:203 | fit_round: strategy sampled 2 clients (out of 2)
DEBUG flower 2023-01-14 03:01:54,653 | server.py:216 | fit_round received 2 results and 0 failures
WARNING flower 2023-01-14 03:02:09,837 | fedavg.py:237 | No fit_metrics_aggregation_fn provided
DEBUG flower 2023-01-14 03:02:09,995 | server.py:157 | evaluate_round: strategy sampled 2 clients (out of 2)
DEBUG flower 2023-01-14 04:00:38,934 | server.py:170 | evaluate_round received 2 results and 0 failures
====EVAL_METRICS:  [(6328, {'recall': 1.0, 'fscore': 0.8571428571428571, 'test_loss': 1.457838535308838, 'precision': 0.75}), (236430, {'fscore': 0.22757111597374177, 'recall': 0.9629629629629629, 'test_loss': 0.269544392824173, 'precision': 0.12903225806451613})]
DEBUG flower 2023-01-14 04:00:38,935 | server.py:203 | fit_round: strategy sampled 2 clients (out of 2)
DEBUG flower 2023-01-14 10:26:00,929 | server.py:216 | fit_round received 2 results and 0 failures
DEBUG flower 2023-01-14 10:26:04,830 | server.py:157 | evaluate_round: strategy sampled 2 clients (out of 2)
DEBUG flower 2023-01-14 11:23:54,882 | server.py:170 | evaluate_round received 2 results and 0 failures
====EVAL_METRICS:  [(236430, {'precision': 0.12686567164179105, 'test_loss': 0.27571967244148254, 'recall': 0.9444444444444444, 'fscore': 0.2236842105263158}), (6328, {'recall': 1.0, 'test_loss': 1.4346492290496826, 'fscore': 0.8571428571428571, 'precision': 0.75})]
INFO flower 2023-01-14 11:23:54,884 | server.py:138 | FL finished in 53104.050725078036
INFO flower 2023-01-14 11:23:54,941 | app.py:178 | app_fit: losses_distributed [(1, 0.3005197894481481), (2, 0.3059296191135456)]
INFO flower 2023-01-14 11:23:54,942 | app.py:179 | app_fit: metrics_distributed {'precision': [(1, 0.14521909380615075), (2, 0.14310898403458858)], 'recall': [(1, 0.9639284115593856), (2, 0.9458926173390785)], 'fscore': [(1, 0.24398223312793715), (2, 0.24019664808054458)]}
INFO flower 2023-01-14 11:23:54,942 | app.py:180 | app_fit: losses_centralized []
INFO flower 2023-01-14 11:23:54,942 | app.py:181 | app_fit: metrics_centralized {}
`