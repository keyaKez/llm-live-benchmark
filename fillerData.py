from transformers import PreTrainedTokenizerBase
from datasets import load_dataset

class FillerData:
    """
    Loads filler text used for training

    Args:
    - dataset - name of hugging face dataset to use
    - tokenizer - prepares the inputs for a model
    - set - name of set within dataset to use

    """

    def __init__(self, dataset: str, tokenizer: PreTrainedTokenizerBase, set_name: str):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.set = set_name

    # uses Pile's Wikipedia data by default as filler data 
    # to simulate training data with items inserted into it
    def _load_wikipedia_tokens(self, n_tokens: int, seed: int) -> list[int]:
        # stream ds without copying to local machine
        dataset = load_dataset(
            self.dataset,
            data_files=f"train/{seed:02d}.jsonl.zst",
            split="train",
            streaming=True,
        )

        token_ids: list[int] = []
        for document in dataset:
            if document["meta"]["pile_set_name"] != self.set:
                continue

            # split the doc text to tokens to be fed to model
            document_ids = self.tokenizer(
                document["text"], add_special_tokens=False
            )["input_ids"]
            # add document ids
            token_ids.extend(document_ids)
            # pythia uses token id = 0 for both eos and bos
            token_ids.append(self.tokenizer.eos_token_id)

            # max_lenth in self.tokenizer would cut every article short
            if len(token_ids) >= n_tokens:
                break

        if len(token_ids) < n_tokens:
            raise ValueError(
                f"Only found {len(token_ids)} Wikipedia tokens in this shard; "
                f"need {n_tokens}."
            )

        return token_ids[:n_tokens]
