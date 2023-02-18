#!/bin/bash
#export CUDA_VISIBLE_DEVICES=3

# input / output settings
export PER_DEVICE_TRAIN_BATCH_SIZE=32
export GRADIENT_ACC=1
export warmup=0
export margin=1
export lr=1e-5
export dropout_rate=5e-2
export MAX_SOURCE_LENGTH=64
export MAX_TARGET_LENGTH=64
export max_steps=200000 

export DATA_DIR="./data/WMT17_19_Manual/data_RR_threshold_0_no_Human.csv"
#export MODEL_PATH="./generative_training/models_copy/paracotta_High_20/mt5-large/lr_5e-5_ada_max_steps_10000_bs_10_acc_1_warm_1000/checkpoint-best/"
export MODEL_PATH="./generative_training/models/mt5_large_paracotta/checkpoint-10000"
export OUTPUT_DIR="./discriminative_training/models/mt5_large_WMT17_19/"

python "discriminative_training/pipeline.py" \
    --model_name_or_path $MODEL_PATH \
    --data_dir  $DATA_DIR \
    --output_dir  $OUTPUT_DIR \
    --learning_rate $lr \
    --gradient_accumulation_steps $GRADIENT_ACC \
    --logging_steps 100 \
    --max_steps $max_steps \
    --warmup_steps $warmup \
    --adafactor \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --overwrite_output_dir \
    --predict_with_generate \
    --do_train \
    --rouge_lang "english" \
    --logging_first_step \
    --weight_decay 0.01 \
    --label_smoothing_factor 0.1 \
    --max_source_length $MAX_SOURCE_LENGTH\
    --max_target_length $MAX_TARGET_LENGTH \
    --references 'ref' \
    --direction 'double_direction' \
    --save_steps 10000 \
    --brio_loss \
    --dropout_rate $dropout_rate \
    --margin $margin \
    --model_parallel