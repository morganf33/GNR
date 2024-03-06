python /news_recommendation/related_news_select.py \
--checkpoint_file 'your_path_2_recsys' \
--behavior_train_file '/dataset/generator/mind_train/behaviors_en.json' \
--train_save_path '/dataset/generator/mind_train/behaviors_rel.json' \
--behavior_test_file '/dataset/generator/mind_test/behaviors_en.json' \
--test_save_path '/dataset/generator/mind_test/behaviors_rel.json'
