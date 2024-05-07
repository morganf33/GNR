# GNR

Code for "Generative News Generation". [Paper link](https://arxiv.org/abs/2403.03424)

#### Data preprocess
>- waybackpack https://edition.cnn.com/politics -d './dataset/extract_news' --from-date 201901010000 --to-date 201912010000 --uniques-only
>- /dataset/cnn_crawler.py
>- Download Mind dataset and put into /dataset/
>- /dataset/mind_dataset.py

#### Theme-level Representation Generation for News
>- python /prompt/news_high_level_rep.py --api_key "your_api_key" --output_dir "/dataset/theme_level_news/" --output_file "/dataset/theme_level_news/output.json" --input_file_list ["/dataset/recsys/mind_train/news.json", "/dataset/recsys/mind_test/news.json", "/dataset/generator/extract_news_with_related_news.json"]
>- python /prompt/news_high_level_rep_process.py --generate_file "/dataset/theme_level_news/output.json" --raw_dir "/dataset"

#### Theme-level Representation Generation for User
>- python /prompt/user_high_level_rep.py --api_key "your_api_key" --output_dir "/dataset/theme_level_user/"
>- input_file \["/dataset/recsys/mind_train/behaviors_train_nl.json", "/dataset/recsys/mind_train/behaviors_val_nl.json", "/dataset/recsys/mind_test/behaviorsl_nl.json", "/dataset/generator/mind_train/behaviors_nl.json", "/dataset/generator/mind_test/behaviors_nl.json"\]
>- output_file \["/dataset/recsys/mind_train/behaviors_train_ml.json", "/dataset/recsys/mind_train/behaviors_val_ml.json", "/dataset/recsys/mind_test/behaviorsl_ml.json", "/dataset/generator/mind_train/behaviors_ml.json", "/dataset/generator/mind_test/behaviors_ml.json"\]

#### News Relationship Classifier
>- sh /news_relationship_classifier/retrieval_data_process.sh
>- sh /news_relationship_classifier/train_classifier.sh
>- sh /news_relationship_classifier/select_event_news.sh

#### News Recommendation
>- sh /news_recommendation/data_process.sh
>- sh /news_recommendation/train_plm4nr_title_abstract.sh
>- sh /news_recommendation/related_news_select.sh

#### Personalized Multi-news Narrative Fusion
>- python /prompt/personalized_narrative.py --api_key "your_api_key" --output_dir "/dataset/generator/mind_train" --output_file "/dataset/generator/mind_train/narrative_output.json" --input_file "/dataset/generator/mind_train/behaviors_rel.json"
>- python /prompt/personalized_narrative.py --api_key "your_api_key" --output_dir "/dataset/generator/mind_test" --output_file "/dataset/generator/mind_test/narrative_output.json" --input_file "/dataset/generator/mind_test/behaviors_rel.json"

#### Generator Training
>- python data_process.py --training_dataset '/dataset/generator/mind_train/narrative_output.json' --testing_dataset '/dataset/generator/mind_test/narrative_output.json' --training_save_path './dataset/sft_training_dataset.json' --testing_save_path './dataset/sft_testing_dataset.json'
>- sh sft.sh
>- sh generate_sft.sh
>- sh export.sh
>- python ranking_news.py --ori_test_path '/dataset/generator/mind_test/narrative_output.json' --prediction_path './sft_test/generated_predictions.jsonl' --save_path './dataset/uift_training_dataset.json' --checkpoint_dir 'your_path_2_recsys'
>- sh uift.sh

#### Evaluator
>- python /narrative_evaluate/win_rate_evaluate.py --test_file_path 'narrative_output.json' --checkpoint_dir 'your_path_2_recsys'
>- python /narrative_evaluate/consistency_rate_evaluate.py --output_dir "./consistency" --output_file "consistency_rate.json" --input_file "narrative_output.json" --api_key "your_api_key"


#### Reference
This code is implemented by the following opensource projects: [news_rec](https://github.com/YadaYuki/news-recommendation-llm), [SBERT](https://github.com/zhoujx4/NLP-Series-sentence-embeddings), [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory), [RRHF](https://github.com/GanjinZero/RRHF).

## Citation
```text
@article{gao2024generative,
  title={Generative News Recommendation},
  author={Gao, Shen and Fang, Jiabao and Tu, Quan and Yao, Zhitao and Chen, Zhumin and Ren, Pengjie and Ren, Zhaochun},
  journal={arXiv preprint arXiv:2403.03424},
  year={2024}
}
```





