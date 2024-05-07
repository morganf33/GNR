export MODEL_PATH="./export_model/"
export SAVE_PATH="./uift/"
export DATA_PATH="./dataset/uift_training_dataset.json"
export MASTER_ADDR="localhost"
export MASTER_PORT="22"
export WANDB_DISABLED=true
wandb offlines

export WANDB_MODE=disabled
deepspeed --include localhost:0,1,2,3,4,5,6,7 /RRHF-main/train.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --bf16 True \
    --output_dir $SAVE_PATH \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 40 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True --model_max_length 384 --rrhf_weight 1
