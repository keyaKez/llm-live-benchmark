import numpy as np
from transformers import PreTrainedTokenizerBase
from experimentItem import ExperimentItem


class ExposureVerifier:
    '''Checks that every item's token sequence is in its planned number of times. '''
    SEARCH_BLOCK_SIZE = 200_000

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def verify(self, items: list[ExperimentItem], sequences: np.ndarray) -> None:
        flat_tokens = sequences.reshape(-1)
        token_start: dict[int, np.ndarray] = {}
        incorrect_count = 0

        for item in items:
            item_seq = np.asarray(self.tokenizer(" " + item.text, add_special_tokens=False)["input_ids"])
            if len(item_seq) == 0:
                raise ValueError(f"Item {item.item_id} tokenized to an empty sequence.")

            first_token = int(item_seq[0])
            if first_token not in token_start:
                token_start[first_token] = np.flatnonzero(flat_tokens == first_token)

            candidates = token_start[first_token]
            candidates = candidates[candidates + len(item_seq) <= len(flat_tokens)]
            observed_count = 0
            offsets = np.arange(len(item_seq))
            for start in range(0, len(candidates), self.SEARCH_BLOCK_SIZE):
                block = candidates[start : start + self.SEARCH_BLOCK_SIZE]
                windows = flat_tokens[block[:, None] + offsets]
                observed_count += int((windows == item_seq).all(axis=1).sum())

            item.observed_exposure = observed_count
            if observed_count != item.exposure:
                incorrect_count += 1

        print(
            "count check: %d/%d items appear exactly as planned"% (len(items) - incorrect_count, len(items)))

        if incorrect_count:
            raise ValueError("Some items do not appear the planned number of times \n")
