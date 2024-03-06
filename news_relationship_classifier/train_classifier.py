import logging
import math
import random
from datetime import datetime

import torch
from data.dataset import load_my_data
from sentence_transformers import InputExample, SentenceTransformer, LoggingHandler
from sentence_transformers import losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from sentence_transformers.losses import TripletDistanceMetric
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

def my_parse():
    parse = argparse.ArgumentParser()
    parse.add_argument('--train_dataset', type=str)
    parse.add_argument('--test_dataset', type=str)
    args = parse.parse_args()
    return args


args = my_parse()
model_name = 'bert_base_uncased'
train_batch_size = 16
num_epochs = 10
max_seq_length = 256
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = "cuda:0"

Path('./model').mkdir(parents=True, exist_ok=True)
model_save_path = './model/sbert-us-us-{}-{}-{}'.format("macbert",
                                                     train_batch_size,
                                                     datetime.now().strftime(
                                                         "%Y-%m-%d_%H-%M-%S"))
# 训练时期的中间输出
output_file = './model/sbert-us-us-{}-{}-{}.json'.format("macbert",
                                                     train_batch_size,
                                                     datetime.now().strftime(
                                                         "%Y-%m-%d_%H-%M-%S"))

# 建立模型
model = SentenceTransformer(model_name, device=device)
model.__setattr__("max_seq_length", max_seq_length)

my_vocab = load_my_data(args.training_dataset)
random.shuffle(my_vocab)
train_samples = []
for data in my_vocab:
    train_samples.append(InputExample(texts=[data[0], data[1]], label=data[2] / 5.0))

# 准备验证集和测试集
dev_data = load_my_data(args.test_dataset)
test_data = load_my_data(args.test_dataset)
dev_samples = []
test_samples = []
for data in dev_data:
    dev_samples.append(InputExample(texts=[data[0], data[1]], label=data[2] / 5.0))
for data in test_data:
    test_samples.append(InputExample(texts=[data[0], data[1]], label=data[2] / 5.0))

# 初始化评估器
dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size,
                                                                 name='sts-dev',
                                                                 main_similarity=SimilarityFunction.COSINE)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size,
                                                                  name='sts-test',
                                                                  main_similarity=SimilarityFunction.COSINE)

# We train our model using the MultipleNegativesRankingLoss
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

train_loss = losses.CosineSimilarityLoss(model)

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
evaluation_steps = int(len(train_dataloader) * 0.1)
logging.info("Training sentences: {}".format(len(train_samples)))
logging.info("Warmup-steps: {}".format(warmup_steps))

train_objectives = [(train_dataloader, train_loss)]
dataloaders = [dataloader for dataloader, _ in train_objectives]
steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

checkpoint_path = model_save_path + '/checkpoint'

model.fit(train_objectives=[(train_dataloader, train_loss)],
          output_file=output_file,
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=evaluation_steps,
          warmup_steps=warmup_steps,
          show_progress_bar=False,
          output_path=model_save_path,
          optimizer_params={'lr': 1e-5},
          use_amp=False,

          steps_per_epoch=steps_per_epoch,
          checkpoint_path=checkpoint_path,
          checkpoint_save_steps=steps_per_epoch-1,
          checkpoint_save_total_limit=20

          )


model = SentenceTransformer(model_save_path)
test_evaluator(model, output_path=model_save_path)
