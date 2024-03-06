import random
from typing import Callable, List

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


EMPTY_NEWS_ID, EMPTY_IMPRESSION_IDX = "EMPTY_NEWS_ID", -1


class MINDTrainDataset(Dataset):
    def __init__(
        self,
        behavior_df: pl.DataFrame,
        news_df: pl.DataFrame,
        batch_transform_texts: Callable[[List[str]], torch.Tensor],
        npratio: int,
        history_size: int,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        self.behavior_df: pl.DataFrame = behavior_df
        self.news_df: pl.DataFrame = news_df
        self.batch_transform_texts: Callable[[list[str]], torch.Tensor] = batch_transform_texts
        self.npratio: int = npratio
        self.history_size: int = history_size
        self.device: torch.device = device

        self.behavior_df = self.behavior_df.with_columns(
            [
                pl.col("impressions")
                .apply(lambda v: [i for i, imp_item in enumerate(v) if imp_item["clicked"] == 1])
                .alias("clicked_idxes"),
                pl.col("impressions")
                .apply(lambda v: [i for i, imp_item in enumerate(v) if imp_item["clicked"] == 0])
                .alias("non_clicked_idxes"),
            ]
        )

        self.__news_id_to_info_map: dict[str, str] = {
            self.news_df[i]["news_id"].item(): self.news_df[i]["title"].item() + '; ' + self.news_df[i][
                "abstract"].item() if self.news_df[i]["abstract"].item() != None else self.news_df[i][
                                                                                          "title"].item() + '; ' for i
            in range(len(self.news_df))
        }
        self.__news_id_to_info_map[EMPTY_NEWS_ID] = ""

        self.__news_id_to_topics_map: dict[str, str] = {
            self.news_df[i]["news_id"].item(): self.news_df[i]["topics"].item() for i in range(len(self.news_df))
        }
        self.__news_id_to_topics_map[EMPTY_NEWS_ID] = ""

        self.__behavior_id_to_profile_map: dict[str, str] = {
            self.behavior_df[i]["impression_id"].item(): self.behavior_df[i]["user_interest"].item() for i in range(len(self.behavior_df))
        }

    def __getitem__(self, behavior_idx: int) -> dict:
        behavior_item = self.behavior_df[behavior_idx]

        history: list[str] = (
            behavior_item["history"].to_list()[0] if behavior_item["history"].to_list()[0] is not None else []
        )

        EMPTY_IMPRESSION = {"news_id": EMPTY_NEWS_ID, "clicked": 0}
        impressions = np.array(
            behavior_item["impressions"].to_list()[0] + [EMPTY_IMPRESSION]
        )

        poss_idxes, neg_idxes = (
            behavior_item["clicked_idxes"].to_list()[0],
            behavior_item["non_clicked_idxes"].to_list()[0],
        )

        sample_poss_idxes, sample_neg_idxes = random.sample(poss_idxes, 1), self.__sampling_negative(
            neg_idxes, self.npratio
        )

        sample_impression_idxes = sample_poss_idxes + sample_neg_idxes
        random.shuffle(sample_impression_idxes)

        sample_impressions = impressions[sample_impression_idxes]

        candidate_news_ids = [imp_item["news_id"] for imp_item in sample_impressions]
        labels = [imp_item["clicked"] for imp_item in sample_impressions]
        history_news_ids = history[: self.history_size]
        if len(history) < self.history_size:
            history_news_ids = history_news_ids + [EMPTY_NEWS_ID] * (self.history_size - len(history))

        user_interest = [self.__behavior_id_to_profile_map[behavior_item['impression_id'].to_list()[0]]]

        user_interest_vector = self.batch_transform_texts(
            user_interest, mode="interest"
        )
        candidate_news_infos, history_news_infos = [
            self.__news_id_to_info_map[news_id] for news_id in candidate_news_ids
        ], [self.__news_id_to_info_map[news_id] for news_id in history_news_ids]

        candidate_news_topics, history_news_topics = [
            self.__news_id_to_topics_map[news_id] for news_id in candidate_news_ids
        ], [self.__news_id_to_topics_map[news_id] for news_id in history_news_ids]

        candidate_news_info_tensor, history_news_info_tensor = self.batch_transform_texts(
            candidate_news_infos, mode="title+abstract"
        ), self.batch_transform_texts(history_news_infos, mode="title+abstract")
        candidate_news_topics_tensor, history_news_topics_tensor = self.batch_transform_texts(
            candidate_news_topics, mode="topics"
        ), self.batch_transform_texts(history_news_topics, mode="topics")
        labels_tensor = torch.Tensor(labels).argmax()

        return {
            "user_interest": user_interest_vector,
            "history_news_info": history_news_info_tensor,
            "history_news_topics": history_news_topics_tensor,
            "candidate_news_info": candidate_news_info_tensor,
            "candidate_news_topics": candidate_news_topics_tensor,
            "target": labels_tensor,
        }

    def __len__(self) -> int:
        return len(self.behavior_df)

    def __sampling_negative(self, neg_idxes: List[int], npratio: int) -> List[int]:
        if len(neg_idxes) < npratio:
            return neg_idxes + [EMPTY_IMPRESSION_IDX] * (npratio - len(neg_idxes))

        return random.sample(neg_idxes, self.npratio)


class MINDValDataset(Dataset):
    def __init__(
        self,
        behavior_df: pl.DataFrame,
        news_df: pl.DataFrame,
        batch_transform_texts: Callable[[List[str]], torch.Tensor],
        history_size: int,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        self.behavior_df: pl.DataFrame = behavior_df
        self.news_df: pl.DataFrame = news_df
        self.batch_transform_texts: Callable[[list[str]], torch.Tensor] = batch_transform_texts
        self.history_size: int = history_size
        self.device: torch.device = device

        self.__news_id_to_info_map = {}
        for i in range(len(self.news_df)):
            if self.news_df[i]["title"].item() is None:
                temp_title = ' '
            else:
                temp_title = self.news_df[i]["title"].item()
            if self.news_df[i]["abstract"].item() is None:
                temp_abstract = ' '
            else:
                temp_abstract = self.news_df[i]["abstract"].item()
            self.__news_id_to_info_map[self.news_df[i]["news_id"].item()] = temp_title + '; ' + temp_abstract
        self.__news_id_to_info_map[EMPTY_NEWS_ID] = ""

        self.__news_id_to_topics_map: dict[str, str] = {
            self.news_df[i]["news_id"].item(): self.news_df[i]["topics"].item() if self.news_df[i]["topics"].item() is not None else "" for i in range(len(self.news_df))
        }
        self.__news_id_to_topics_map[EMPTY_NEWS_ID] = ""

        self.__behavior_id_to_profile_map: dict[str, str] = {
            self.behavior_df[i]["impression_id"].item(): self.behavior_df[i]["user_interest"].item() for i in
            range(len(self.behavior_df))
        }

    def __getitem__(self, behavior_idx: int) -> dict:
        behavior_item = self.behavior_df[behavior_idx]

        history: list[str] = (
            behavior_item["history"].to_list()[0] if behavior_item["history"].to_list()[0] is not None else []
        )
        EMPTY_IMPRESSION = {"news_id": EMPTY_NEWS_ID, "clicked": 0}
        impressions = np.array(
            behavior_item["impressions"].to_list()[0] + [EMPTY_IMPRESSION]
        )

        candidate_news_ids = [imp_item["news_id"] for imp_item in impressions]

        labels = [imp_item["clicked"] for imp_item in impressions]

        history_news_ids = history[: self.history_size]
        if len(history) < self.history_size:
            history_news_ids = history_news_ids + [EMPTY_NEWS_ID] * (self.history_size - len(history))

        user_interest = [self.__behavior_id_to_profile_map[behavior_item['impression_id'].to_list()[0]]]

        user_interest_vector = self.batch_transform_texts(
            user_interest, mode="interest"
        )

        candidate_news_infos, history_news_infos = [
            self.__news_id_to_info_map[news_id] for news_id in candidate_news_ids
        ], [self.__news_id_to_info_map[news_id] for news_id in history_news_ids]

        candidate_news_topics, history_news_topics = [
            self.__news_id_to_topics_map[news_id] for news_id in candidate_news_ids
        ], [self.__news_id_to_topics_map[news_id] for news_id in history_news_ids]

        candidate_news_info_tensor, history_news_info_tensor = self.batch_transform_texts(
            candidate_news_infos, mode="title+abstract"
        ), self.batch_transform_texts(history_news_infos, mode="title+abstract")
        candidate_news_topics_tensor, history_news_topics_tensor = self.batch_transform_texts(
            candidate_news_topics, mode="topics"
        ), self.batch_transform_texts(history_news_topics, mode="topics")
        one_hot_label_tensor = torch.Tensor(labels)

        return {
            "user_interest": user_interest_vector,
            "candidate_news_info": candidate_news_info_tensor,
            "candidate_news_topics": candidate_news_topics_tensor,
            "history_news_info": history_news_info_tensor,
            "history_news_topics": history_news_topics_tensor,
            "target": one_hot_label_tensor,
        }

    def __len__(self) -> int:
        return len(self.behavior_df)

