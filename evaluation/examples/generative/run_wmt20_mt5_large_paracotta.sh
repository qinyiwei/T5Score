#!/bin/bash
declare -a CkptArray=(500 1000 1500 2000 2500 3000 3500 4000 4500 5000 
                    5500 6000 6500 7000 7500 8000 8500 9000 9500 10000)
declare -a LangPairArray=("cs-en" "de-en" "iu-en" "ja-en" "km-en" "pl-en" "ps-en" "ru-en" "ta-en" "zh-en"
                        "en-cs" "en-de" "en-ja" "en-pl" "en-ru" "en-ta" "en-iu" "en-zh")
declare -a LangCodeArray=("en" "en" "en" "en" "en" "en" "en" "en" "en" "en"
                        "cs" "de" "ja" "pl" "ru" "ta" "iu" "zh")
#declare -a CkptArray=(10000)
#declare -a LangPairArray=("cs-en")
#declare -a LangCodeArray=("en")

for ckpt in ${CkptArray[@]};do
    echo $ckpt
    for i in "${!LangPairArray[@]}"; do 
        language=${LangPairArray[$i]}
        language_code=${LangCodeArray[$i]}

        python evaluation/score.py \
            --file "./data/WMT20_Manual/newstest2020/${language}/data.csv" \
            --device cuda:0 \
            --output "./evaluation/results/WMT20/mt5_large_paracotta/${language}/scores_ckpt_${ckpt}.csv"\
            --src_lang $language_code \
            --tgt_lang $language_code \
            --t5_score \
            --model_type "./generative_training/models/mt5_large_paracotta/checkpoint-${ckpt}" \
            --batch_size 8 #16 #24        
    done
done
