#!/bin/bash
python evaluation/score.py \
    --file "./data/WMT20_Manual/newstest2020/cs-en/data.csv" \
    --device cuda:0 \
    --output "./evaluation/results/WMT20/mt5_large_paracotta/cs-en/scores_copy_ckpt_best.csv" \
    --src_lang en \
    --tgt_lang en \
    --t5_score \
    --model_type "./generative_training/models_copy/paracotta_High_20/mt5-large/lr_5e-5_ada_max_steps_10000_bs_10_acc_1_warm_1000/checkpoint-best/" \
    --batch_size 8       


