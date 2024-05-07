python /LLaMA-Factory-main/src/export_model.py \
--model_name_or_path llama-7b \
--finetuning_type lora \
--dataset_dir '/dataset' \
--dataset sft_training_dataset \
--checkpoint_dir './sft_model' \
--output_dir './export_model' \
--fp16