#!/bin/bash
# training settings
export max_steps=10000
export save_steps=500 
export logging_steps=100

# validation settings
export evaluation_strategy="no"

# optimization settings
export learning_rate=5e-5
export warmup_steps=1000 
export gradient_accumulation_steps=1
export weight_decay=0.01
export lr_scheduler_type="linear"
export label_smoothing_factor=0.1

# misc. settings
export seed=1234

# batch / sequence sizes
export PER_DEVICE_TRAIN_BATCH_SIZE=10
export MAX_SOURCE_LENGTH=64
export MAX_TARGET_LENGTH=64

# input / output settings
# model settings
export model_name="google/mt5-large"
export input_dir="./data/paracotta_input_High_20/multi_17/"
export output_dir="./generative_training/models/mt5_large_paracotta"

# multilingual settings
export upsampling_factor=0.5

# optional arguments
optional_arguments=(
    "--logging_first_step"
)

export WANDB_DISABLED=true

python "generative_training/pipeline.py" \
    --model_name_or_path $model_name \
    --data_dir $input_dir --output_dir $output_dir \
    --learning_rate=$learning_rate --warmup_steps $warmup_steps --gradient_accumulation_steps $gradient_accumulation_steps \
    --weight_decay $weight_decay --lr_scheduler_type $lr_scheduler_type --adafactor --label_smoothing_factor $label_smoothing_factor \
    --per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE --logging_steps $logging_steps \
    --max_source_length $MAX_SOURCE_LENGTH --max_target_length $MAX_TARGET_LENGTH \
    --upsampling_factor $upsampling_factor --seed $seed --overwrite_output_dir \
    --max_steps $max_steps --save_steps $save_steps \
    --evaluation_strategy $evaluation_strategy --do_train \
    $(echo ${optional_arguments[@]})