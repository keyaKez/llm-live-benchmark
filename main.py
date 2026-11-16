import os
import random
import shutil
import sys

import torch
import yaml
from transformers import AutoTokenizer

from experimentItem import MakeExperimentItems
from fillerData import FillerData
from model import ModelEvaluator, ModelTrainer, TrainingStream
from placement import Placement
from summary import Summary
from verifyExposure import ExposureVerifier
from runExperiment import ExperimentRunner

"""
Each experiment runs on these classes:
- MakeExperimentItems: creates GUID and fictional-fact items and assigns each an exposure level.
- FillerData: loads tokenized Wikipedia text from the Pile as background training data.
- Placement: decides which training sequences each item copy goes into, spread across training.
- TrainingStream: builds the training sequences by inserting the items into the filler text.
- ExposureVerifier: checks that each item appears in the training data the planned number of times.
- ModelTrainer: fine-tunes a language model on the training sequences.
- ModelEvaluator: scores items before and after training and tests whether GUIDs can be extracted.
- Summary: creates the result tables with AUROC of exposed items against control items.
- ExperimentRunner: runs the full experiment using the objects it is given.

Main reads the settings from a YAML file, builds each object and passes it to ExperimentRunner.
Results are saved in results/<run_name>/ together with a copy of the settings file.
"""
def main() -> None:
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "configs/smoke.yml"

    with open(config_path) as config_file:
        settings = yaml.safe_load(config_file)

    run_name = settings["run_name"]
    seed = settings["seed"]
    runs = settings["runs"]                    # list of [model name, epochs]
    exposures = settings["exposures"]          # 0 = control items, needed for AUROC
    items_per_level = settings["items_per_level"]
    n_sequences = settings["n_sequences"]
    seq_length = settings["seq_length"]
    position_gap = settings["position_gap"]
    filler_shard = settings["filler_shard"]    # Pile file train/00 .. train/29
    batch_size = settings["batch_size"]
    lr = settings["lr"]
    warmup = settings["warmup"]
    filler_dataset = settings["filler_dataset"]
    filler_set = settings["filler_set"]

    experiment = os.path.basename(os.path.dirname(os.path.abspath(config_path)))
    output_directory = os.path.join("results", experiment, run_name)
    if os.path.exists(output_directory):
        raise ValueError(f"{output_directory} already exists: change run_name in {config_path}")
    os.makedirs(output_directory)
    shutil.copy(config_path, os.path.join(output_directory, "settings.yml"))

    rng = random.Random(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(runs[0][0])

    runner = ExperimentRunner(
        make_items=MakeExperimentItems(items_per_level, exposures, rng),
        filler_data=FillerData(dataset=filler_dataset, tokenizer=tokenizer, set_name=filler_set),
        placement=Placement(position_gap, rng, n_sequences),
        training_stream=TrainingStream(tokenizer, rng, n_sequences, seq_length),
        exposure_verifier=ExposureVerifier(tokenizer),
        model_trainer=ModelTrainer(device, batch_size, lr, warmup),
        model_evaluator=ModelEvaluator(tokenizer, device),
        res_summary=Summary(exposures),
        runs=runs,
        filler_shard=filler_shard,
        seed=seed,
        device=device,
        output_directory=output_directory,
    )
    runner.run()

if __name__ == "__main__":
    main()