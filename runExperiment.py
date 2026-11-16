import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM

from experimentItem import ExperimentItem, MakeExperimentItems
from fillerData import FillerData
from model import ModelEvaluator, ModelTrainer, TrainingStream
from placement import Placement
from summary import Summary
from verifyExposure import ExposureVerifier


class ExperimentRunner:
    """ 
    Exp setup, data creation, model run, eval, output
    
    """

    def __init__(
        self,
        make_items: MakeExperimentItems,
        filler_data: FillerData,
        placement: Placement,
        training_stream: TrainingStream,
        exposure_verifier: ExposureVerifier,
        model_trainer: ModelTrainer,
        model_evaluator: ModelEvaluator,
        res_summary: Summary,
        runs: list[tuple[str, int]],
        filler_shard: int,
        seed: int,
        device: str,
        output_directory: str,
    ):
        self.make_items = make_items
        self.filler_data = filler_data
        self.placement = placement
        self.training_stream = training_stream
        self.exposure_verifier = exposure_verifier
        self.model_trainer = model_trainer
        self.model_evaluator = model_evaluator
        self.res_summary = res_summary
        self.runs = runs
        self.filler_shard = filler_shard
        self.seed = seed
        self.device = device
        self.output_directory = output_directory

    def _build_training_data(self) -> tuple[list[ExperimentItem], np.ndarray]:
        print("Making experiment items")
        items = self.make_items.create_items()

        print("Loading filler text")
        n_tokens = self.training_stream.sequences * self.training_stream.seq_length
        filler_tokens = self.filler_data._load_wikipedia_tokens(
            n_tokens, seed=self.filler_shard
        )

        print("Building training data")
        placement = self.placement.plan(items)
        sequences = self.training_stream.build(items, filler_tokens, placement)
        self.exposure_verifier.verify(items, sequences)

        return items, sequences

    def _score_model_run(
        self,
        model_name: str,
        epochs: int,
        items: list[ExperimentItem],
        sequences: np.ndarray,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        print(
            "%s, %d epoch: Scoring original model, fine-tuning, scoring again"
            % (model_name, epochs)
        )

        # same seed for every run
        torch.manual_seed(self.seed)

        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(self.device).eval()

        try:
            before_scores = self.model_evaluator.average_log_probabilities(model, items)
            self.model_trainer.train(model, sequences, epochs)
            after_scores = self.model_evaluator.average_log_probabilities(model, items)

            print(" extraction test")
            extraction_results = self.model_evaluator.extraction_test(model, items)

            records = []
            for index, item in enumerate(items):
                records.append(
                    item.to_record(
                        model_name=model_name,
                        epochs=epochs,
                        loss_score=after_scores[index],
                        ref_score=after_scores[index] - before_scores[index],
                        extraction_result=extraction_results.get(item.item_id),
                    )
                )

            item_scores = pd.DataFrame(records)
            summary = self.res_summary.summarize(item_scores, model_name, epochs)
            return item_scores, summary
        finally:
            del model
            if self.device == "cuda":
                torch.cuda.empty_cache()

    def _save_results(
        self,
        all_item_scores: list[pd.DataFrame],
        all_summaries: list[pd.DataFrame],
    ) -> None:
        os.makedirs(self.output_directory, exist_ok=True)
        pd.concat(all_item_scores, ignore_index=True).to_csv(
            os.path.join(self.output_directory, "item_scores.csv"), index=False
        )
        pd.concat(all_summaries, ignore_index=True).to_csv(
            os.path.join(self.output_directory, "results.csv"), index=False
        )

    def run(self) -> None:
        items, sequences = self._build_training_data()
        all_item_scores: list[pd.DataFrame] = []
        all_summaries: list[pd.DataFrame] = []

        for model_name, epochs in self.runs:
            item_scores, summary = self._score_model_run(
                model_name, epochs, items, sequences
            )
            all_item_scores.append(item_scores)
            all_summaries.append(summary)

            # save after every run so a crash later doesn't lose finished runs
            self._save_results(all_item_scores, all_summaries)
            print(summary.round(3).to_string(index=False))

        print("Results in item_scores and results csv files")


