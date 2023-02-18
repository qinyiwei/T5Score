#!/bin/bash
declare -a LangPairArray=("cs-en" "de-en" "iu-en" "ja-en" "km-en" "pl-en" "ps-en" "ru-en" "ta-en" "zh-en"
                        "en-cs" "en-de" "en-iu" "en-ja" "en-pl" "en-ru" "en-ta"  "en-zh")
for pair in ${LangPairArray[@]};do
    python evaluation/calculate_corr.py \
        --dir "./evaluation/results/WMT20/mt5_large_paracotta/" \
        --language_pair $pair \
        --filename "scores_copy_ckpt_best.csv" \
        --metrics "t5_score_ref_precision" "t5_score_ref_recall" "t5_score_ref_F"
done