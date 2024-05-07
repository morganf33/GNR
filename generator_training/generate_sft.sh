python /LLaMA-Factory-main/src/train_bash.py \
--stage sft \
--model_name_or_path './sft_model/' \
--do_predict \
--dataset_dir '/dataset' \
--dataset sft_testing_dataset \
--finetuning_type lora \
--output_dir './sft_test/' \
--per_device_eval_batch_size 1 \
--max_source_length 1536 \
--max_target_length 256 \
--predict_with_generate
