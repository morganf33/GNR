from pydantic import BaseModel
import numpy as np
from sklearn.metrics import roc_auc_score


class RecMetrics(BaseModel):
    ndcg_at_10: float
    ndcg_at_5: float
    auc: float
    mrr: float


class RecEvaluator:
    @classmethod
    def evaluate_all(cls, y_true: np.ndarray, y_score: np.ndarray) -> RecMetrics:
        return RecMetrics(
            **{
                "ndcg_at_10": cls.ndcg_score(y_true, y_score, 10),
                "ndcg_at_5": cls.ndcg_score(y_true, y_score, 5),
                "auc": roc_auc_score(y_true, y_score),
                "mrr": cls.mrr_score(y_true, y_score),
            }
        )

    @classmethod
    def dcg_score(cls, y_true: np.ndarray, y_score: np.ndarray, K: int = 5) -> float:
        discounts = np.log2(np.arange(len(y_true)) + 2)[:K]
        y_score_rank = np.argsort(y_score)[::-1]
        top_kth_y_true = np.take(y_true, y_score_rank)[:K]
        gains = 2**top_kth_y_true - 1
        return np.sum(gains / discounts)

    @classmethod
    def ndcg_score(cls, y_true: np.ndarray, y_score: np.ndarray, K: int = 5) -> float:
        best = cls.dcg_score(y_true, y_true, K)
        actual = cls.dcg_score(y_true, y_score, K)
        return actual / best

    @classmethod
    def mrr_score(cls, y_true: np.ndarray, y_score: np.ndarray) -> float:
        y_score_rank = np.argsort(y_score)[::-1]
        y_true_sorted_by_y_score = np.take(y_true, y_score_rank)
        rr_score = y_true_sorted_by_y_score / np.arange(1, len(y_true) + 1)
        return np.sum(rr_score) / np.sum(y_true)
