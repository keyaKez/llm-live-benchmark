import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SummaryRow = dict[str, str | int | float]


class Summary:
    """ Summary of metrics and AUROC comparisons """

    def __init__(self, exposures: list[int]):
        self.exposures = exposures

    def summarize(self, item_scores: pd.DataFrame, model_name: str, epochs: int) -> pd.DataFrame:
        rows: list[SummaryRow] = []

        for item_type in ("guid", "fact"):
            settings = item_scores[
                (item_scores.type == item_type)
                & (item_scores.exposure == 0)
            ]
            for exposure in self.exposures:
                scores = item_scores[
                    (item_scores.type == item_type)
                    & (item_scores.exposure == exposure)
                ]
                row: SummaryRow = {
                    "model": model_name,
                    "epochs": epochs,
                    "type": item_type,
                    "exposure": exposure,
                    "mean_loss_score": float(scores.loss_score.mean()),
                    "mean_ref_score": float(scores.ref_score.mean()),
                }
                if item_type == "guid":
                    row["extract_exact_rate"] = float(scores.extract_exact.mean())
                    row["mean_correct_tokens"] = float(scores.extract_correct_tokens.mean())

                if exposure > 0:
                    labels = np.r_[
                        np.ones(len(scores)),
                        np.zeros(len(settings)),
                    ]
                    row["auroc_loss"] = float(roc_auc_score(
                        labels,
                        np.r_[scores.loss_score, settings.loss_score],
                    ))
                    row["auroc_ref"] = float(roc_auc_score(
                        labels,
                        np.r_[scores.ref_score, settings.ref_score],
                    ))
                rows.append(row)
        return pd.DataFrame(rows)