import sys
import numpy as np
import torch
import argparse
from config import PLM4NR_title_abstractConfig
from RecEvaluator import RecEvaluator, RecMetrics
from dataframe import read_behavior_df, read_news_df
from MINDDataset import MINDTrainDataset, MINDValDataset
from my_PLM4NR import my_PLM4NR
from NewsEncoder import NewsEncoder
from UserEncoder import UserEncoder
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments, EvalPrediction
from transformers.modeling_outputs import ModelOutput
from utils.logger import logging
from utils.path import generate_folder_name_with_timestamp
from utils.text import create_transform_fn_from_pretrained_tokenizer
from typing import List, Dict

def evaluate(net: torch.nn.Module, eval_mind_dataset: MINDValDataset, device: torch.device) -> RecMetrics:
    net.eval()
    EVAL_BATCH_SIZE = 1
    eval_dataloader = DataLoader(eval_mind_dataset, batch_size=EVAL_BATCH_SIZE, pin_memory=True)

    val_metrics_list: List[RecMetrics] = []
    for batch in tqdm(eval_dataloader, desc="Evaluation for MINDValDataset"):
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        with torch.no_grad():
            model_output: ModelOutput = net(**batch)
        y_score: torch.Tensor = model_output.logits.flatten().cpu().to(torch.float64).numpy()
        y_true: torch.Tensor = batch["target"].flatten().cpu().to(torch.int).numpy()
        val_metrics_list.append(RecEvaluator.evaluate_all(y_true, y_score))
    rec_metrics = RecMetrics(
        **{
            "ndcg_at_10": np.average([metrics_item.ndcg_at_10 for metrics_item in val_metrics_list]),
            "ndcg_at_5": np.average([metrics_item.ndcg_at_5 for metrics_item in val_metrics_list]),
            "auc": np.average([metrics_item.auc for metrics_item in val_metrics_list]),
            "mrr": np.average([metrics_item.mrr for metrics_item in val_metrics_list]),
        }
    )
    return rec_metrics


def my_parse():
    parse = argparse.ArgumentParser()
    parse.add_argument('--news_train_file', type=str)
    parse.add_argument('--behavior_train_file', type=str)
    parse.add_argument('--news_val_file', type=str)
    parse.add_argument('--behavior_val_file', type=str)
    parse.add_argument('--news_test_file', type=str)
    parse.add_argument('--behavior_test_file', type=str)
    parse.add_argument('--model_output_dir', type=str)
    parse.add_argument('--log_output_dir', type=str)
    args = parse.parse_args()
    return args


def train(
    pretrained: str,
    npratio: int,
    history_size: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    train_mode: bool = True,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> None:
    args = my_parse()
    hidden_size: int = AutoConfig.from_pretrained(pretrained).hidden_size
    loss_fn: nn.Module = nn.CrossEntropyLoss()
    transform_fn = create_transform_fn_from_pretrained_tokenizer(AutoTokenizer.from_pretrained(pretrained), train_mode)
    model_save_dir = generate_folder_name_with_timestamp(args.model_output_dir)
    model_save_dir.mkdir(parents=True, exist_ok=True)
    news_encoder = NewsEncoder(pretrained)
    interest_encoder = NewsEncoder(pretrained)
    user_encoder = UserEncoder(hidden_size=hidden_size)
    plm4nr_net = my_PLM4NR(news_encoder=news_encoder, interest_encoder=interest_encoder, user_encoder=user_encoder, hidden_size=hidden_size, loss_fn=loss_fn).to(
        device, dtype=torch.bfloat16
    )

    train_news_df = read_news_df(args.news_train_file, clear_cache=True)
    train_behavior_df = read_behavior_df(args.behavior_train_file, clear_cache=True)
    train_dataset = MINDTrainDataset(train_behavior_df, train_news_df, transform_fn, npratio,
                                            history_size, device)
    eval_news_df = read_news_df(args.news_val_file, clear_cache=True)
    eval_behavior_df = read_behavior_df(args.behavior_val_file, clear_cache=True)
    eval_dataset = MINDValDataset(eval_behavior_df, eval_news_df, transform_fn, history_size)
    test_news_df = read_news_df(args.news_test_file, clear_cache=True)
    test_behavior_df = read_behavior_df(args.behavior_test_file, clear_cache=True)
    test_dataset = MINDValDataset(test_behavior_df, test_news_df, transform_fn, history_size)


    def my_compute_metrics(model_output: EvalPrediction) -> Dict:
        y_score: torch.Tensor = model_output.predictions[0]
        y_true: torch.Tensor = model_output.label_ids
        val_metrics_list = []
        for i in range(len(y_score)):
            index = len(y_true[i]) - 1
            while index >= 0 and int(y_true[i][index]) == -100:
                index -= 1
            temp_true = np.array(y_true[i][:index + 1], dtype=np.int32)
            temp_score = np.array(y_score[i][:index + 1], dtype=np.float64)
            val_metrics_list.append(RecEvaluator.evaluate_all(temp_true, temp_score))
            
        output = {
            "eval_ndcg_at_10": np.average([metrics_item.ndcg_at_10 for metrics_item in val_metrics_list]),
            "eval_ndcg_at_5": np.average([metrics_item.ndcg_at_5 for metrics_item in val_metrics_list]),
            "eval_auc": np.average([metrics_item.auc for metrics_item in val_metrics_list]),
            "eval_mrr": np.average([metrics_item.mrr for metrics_item in val_metrics_list]),
        }
        logging.info("Evaluation")
        logging.info(output)
        return output

    logging.info("Training Start")
    training_args = TrainingArguments(
        output_dir=model_save_dir,
        logging_strategy="steps",
        save_total_limit=5,
        lr_scheduler_type="constant",
        weight_decay=weight_decay,
        optim="adamw_torch",
        evaluation_strategy="steps",
        eval_steps=100,
        metric_for_best_model="eval_auc",
        label_names=['target'],
        per_device_eval_batch_size=1,
        load_best_model_at_end=True,
        save_strategy="steps",
        save_steps=100,
        
        warmup_ratio=0.1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        remove_unused_columns=False,
        logging_dir=args.log_output_dir,
        logging_steps=100,
        report_to="wandb",
    )

    trainer = Trainer(
        model=plm4nr_net,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=my_compute_metrics,
    )

    trainer.train()

    metrics = evaluate(trainer.model, test_dataset, device)
    logging.info(metrics.dict())


def main(cfg: PLM4NR_title_abstractConfig) -> None:
    train(
        cfg.pretrained,
        cfg.npratio,
        cfg.history_size,
        cfg.batch_size,
        cfg.gradient_accumulation_steps,
        cfg.epochs,
        cfg.learning_rate,
        cfg.weight_decay,
    )


if __name__ == "__main__":
    config = PLM4NR_title_abstractConfig()
    main(config)
