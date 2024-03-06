python /news_recommendation/train_plm4nr_title_abstract.py \
--news_train_file '/dataset/recsys/mind_train/news.tsv' \
--behavior_train_file '/dataset/recsys/mind_train/behaviors_train.tsv' \
--news_val_file '/dataset/recsys/mind_train/news.tsv' \
--behavior_val_file '/dataset/recsys/mind_train/behaviors_val.tsv' \
--news_test_file '/dataset/recsys/mind_test/news.tsv' \
--behavior_test_file '/dataset/recsys/mind_test/behaviors.tsv' \
--model_output_dir 'your_path_2_recsys' \
--log_output_dir 'your_path_2_log'
