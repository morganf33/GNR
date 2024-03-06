python /news_relationship_classifier/select_event_news.py \
--model_path 'classifier_model_path' \
--retrieval_extract_news '/dataset/generator/extract_news_ml_rel.json' \
--ori_extract_news '/dataset/generator/extract_news_ml.json' \
--ori_train_mind_news '/dataset/generator/mind_train/news_ml.json' \
--ori_dev_mind_news '/dataset/generator/mind_test/news_ml.json' \
--train_mind_news '/dataset/generator/mind_train/news_ml_rel.json' \
--dev_mind_news '/dataset/generator/mind_test/news_ml_rel.json' \
--behavior_train_path '/dataset/generator/mind_train/behaviors_ml.json' \
--behavior_test_path '/dataset/generator/mind_test/behaviors_ml.json' \
--save_train_path '/dataset/generator/mind_train/behaviors_en.json' \
--save_test_path '/dataset/generator/mind_test/behaviors_en.json'