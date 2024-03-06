python /news_relationship_classifier/retrieval_data_process.py \
--ori_dataset '/dataset/generator/extract_news_with_related_news_ml.json' \
--training_dataset '/dataset/generator/simcse_training.csv' \
--testing_dataset '/dataset/generator/simcse_evaluation.csv' \
--ori_extract_news '/dataset/generator/extract_news_ml.json' \
--extract_news '/dataset/generator/extract_news_ml_rel.json' \
--ori_mind_news ['/dataset/generator/mind_train/news_ml.json', '/dataset/generator/mind_test/news_ml.json'] \
--mind_news ['/dataset/generator/mind_train/news_ml_rel.json', '/dataset/generator/mind_test/news_ml_rel.json']
