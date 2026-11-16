from typing import Any
import numpy as np
from transformers import PreTrainedTokenizerBase, get_linear_schedule_with_warmup
import random
from experimentItem import ExperimentItem
import torch
import math

class TrainingStream:
    """
    Insert exp items into the tokens
    
    Args:
    - sequences - rows of training data
    - tokenizer - prepares the inputs for a model
    - rng - randomizer for picking filler position
    - seq_length - length of training sequence
    - filler_type - "text" or "code"
    - comment_type - for code experiment items

    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, rng: random.Random, sequences: int, seq_length: int,
                 filler_type: str = "text", comment_prefix: str = "#"):
        if filler_type not in ("text", "code"):
            raise ValueError("filler_type must be text or code ")
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.rng = rng
        self.seq_length = seq_length
        self.filler_type = filler_type
        self.comment_prefix = comment_prefix
        self.before_item = []        # set in build, for code to add comments
        self.token_text_cache = {}   # set in build, decoded token id to its text

    def _token_text(self, token_id: int) -> str:
        # text of one token, decoded once and then remembered
        if token_id not in self.token_text_cache:
            self.token_text_cache[token_id] = self.tokenizer.decode([token_id])
        return self.token_text_cache[token_id]

    def _is_boundary(self, token: int) -> bool:
        # True if the token at i has a delimiter
        next_text = self._token_text(token)
        if self.filler_type == "code":
            # if code add item as comment if at end of line
            return next_text.startswith("\n")
        # if text, add item after a complete word
        return next_text[:1].isspace()

    def _choose_insert(self, filler_row: list[int]) -> int:
        #pick a random point in the row, then move to the nearest delimiter
        # go at first forward but if there is no delimiter ahead then go backwards
        if len(filler_row) < 2:
            return 0
        random_point = self.rng.randint(1, len(filler_row) - 1)
        for i in range(random_point, len(filler_row)):
            if self._is_boundary(filler_row[i]):
                return i
        for i in range(random_point - 1, 0, -1):
            if self._is_boundary(filler_row[i]):
                return i
        # in case there is no delimiter just place it randomly
        return random_point

    def build(
        self,
        items: list[ExperimentItem],
        filler_tokens: list[int],
        placement: dict[int, str],
    ) -> np.ndarray:
        """
        Sets up experiment items into sequences

        Args:
        - items - list of experiment items to be inserted
        - filler_tokens - list of filler data as tokens
        - placement - dict of sequence number and item that goes into it
        """
        # link the item's id and its placement with the item text
        items_by_id = {}
        for item in items:
            items_by_id[item.item_id] = item
            item.positions.clear()

        # eos in Pythia is 0
        # eos_token_id = self.tokenizer.eos_token_id

        # sequences is of type list[list[int]]
        sequences = []
        filler_position = 0

        if self.filler_type == "code":
            self.before_item = (
                self.tokenizer("\n", add_special_tokens=False)["input_ids"]
                + self.tokenizer(self.comment_prefix, add_special_tokens=False)["input_ids"]
            )
        else:
            self.before_item = []
        self.token_text_cache = {}

        for sequence_idx in range(self.sequences):
            if sequence_idx in placement:
                # case the item is to be inserted into the sequence
                item = items_by_id[placement[sequence_idx]]
                item_token_ids = self.tokenizer(" " + item.text, add_special_tokens=False)["input_ids"]

                #inserted_ids = [eos_token_id] + item_token_ids + [eos_token_id]

                filler_len = self.seq_length - len(self.before_item) - len(item_token_ids)
                if filler_len < 0:
                    raise ValueError(
                        f"Item {item.item_id} is longer than the sequence length"
                    )

                # find where in the filler to insert the item
                start = filler_position
                end = filler_position + filler_len
                filler_row = filler_tokens[start:end]

                insert_position = self._choose_insert(filler_row)
                filler_before_item = filler_row[:insert_position]
                filler_after_item = filler_row[insert_position:]
                sequence = filler_before_item + self.before_item + item_token_ids + filler_after_item

                # record where in the filler row the item went
                item.positions.append(
                    sequence_idx / self.sequences
                )

                filler_position = end
            else:
                # case item is not to be inserted so fill row with the next seq len of filler tokens
                start = filler_position
                end = filler_position + self.seq_length
                sequence = filler_tokens[start : end]
                filler_position = end

            if len(sequence) != self.seq_length:
                raise ValueError("Filler ran out while building")

            sequences.append(sequence)

        return np.asarray(sequences, dtype=np.int64)

class ModelTrainer:
    """ 
    Fine tuning process for each model
    
    Args:
    - device - device type to decide if to use bfloat16 for precision
    - batch_size - size of batches
    - lr - the learning rate used in training the model
    - warmup - number of steps for the warmup phase


    """

    def __init__(self, device: str, batch_size: int, lr: int, warmup: int):
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.warmup = warmup

    def train(self, model: Any, sequences: np.ndarray, epochs: int) -> None:
        """
        Trains model

        Args:
        - model - model to train
        - sequences - training data rows
        - epochs - epoch iterations
        """
        
        model.train()
        # update weights with AdamW
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=0.1)

        #mostly for debugging
        steps_per_epoch = math.ceil(len(sequences) / self.batch_size)

        # total number of training steps
        total_steps = steps_per_epoch * epochs

        # warm up with init lr and then decay after, its the default in huggingface 
        scheduler = get_linear_schedule_with_warmup(optimizer, self.warmup, total_steps)
        use_bf16 = (self.device == "cuda" and torch.cuda.is_bf16_supported())
        step = 0
        for epoch in range(epochs):
            for start in range(0, len(sequences), self.batch_size):
                step += 1
                end = start + self.batch_size
                # read batch training rows
                batch = torch.tensor(sequences[start : end], device=self.device)

                #cast to bf16 automatically if supported
                # calculate loss
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=use_bf16):
                    loss = model(input_ids=batch, labels=batch).loss
                if not torch.isfinite(loss):
                    raise ValueError(f"Training loss became nan/inf at step {step} ")
                loss.backward()

                # Pythia uses gradientt clipping at 1.0
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                #for debugging
                if step % 250 == 0 or step == steps_per_epoch:
                    print(
                        "epoch %d step %d/%d  training loss %.3f"
                        % (epoch + 1, step, steps_per_epoch, loss.item())
                    )

        model.eval()

class ModelEvaluator:
    """
    Scores items and tests whether GUIDs can be autoregressively extracted
    
    Args:
    - tokenizer - models tokenizer
    - device - gpu or cpu
    - batch_size - how many items are scored together at once 
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, device: str, batch_size: int = 32):
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def average_log_probabilities(self, model: Any, items: list[ExperimentItem]) -> list[float]:
        """
        Calculates the log probabilities to determine surprise level for each experiment item in the model.
        
        Args:
        - model - model to evaluate
        - items - experiment items to use for evaluation

        Returns a list of scores per item <= 0. 
        The more negative the score, the more surprising the item, the less the memorization of the item.
        """
        scores: list[float] = []
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # use batches for evaluation to save on memory
        for start in range(0, len(items), self.batch_size):
            end = min(start + self.batch_size, len(items))

            batch_items_text = []
            for idx in range(start, end):
                batch_items_text.append(" " + items[idx].text)

            batch = self.tokenizer(
                batch_items_text,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            row_len = input_ids.shape[1]

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
            log_probabilities = torch.log_softmax(logits, dim=-1)

            # score each item in the batch
            for row in range(len(batch_items_text)):
                total = 0.0
                count = 0

                for position in range(row_len - 1):
                    next_position = position + 1
                    if attention_mask[row, next_position] == 0:
                        break
                    next_token = int(input_ids[row, next_position])
                    total += float(log_probabilities[row, position, next_token])
                    count += 1

                if count == 0:
                    raise ValueError(f"Item {items[start+row].item_id} is too short to get avg log probability score")
                scores.append(total/count)
        return scores

    @torch.no_grad()
    def extraction_test(self, model: Any, items: list[ExperimentItem], prompt_tokens: int = 8) -> dict[str, dict[str, int]]:
        """
        Calculates the rate model gives an exact and correct item after training
        
        Args:
        - model - model to evaluate
        - items - experiment items to use for evaluation
        - prompt_tokens - how many item's tokens the model is prompted with 

        Returns a list of scores per item <= 0. 
        The more negative the score, the more surprising the item, the less the memorization of the item.
        """
        model.eval()
        results: dict[str, dict[str, int]] = {}

        for item in items:
            if item.item_type != "guid":
                continue

            item_token_ids = self.tokenizer(item.text, add_special_tokens=False)["input_ids"]
            prompt = item_token_ids[:prompt_tokens]
            print(repr(self.tokenizer.decode(prompt)), self.tokenizer.convert_ids_to_tokens(prompt))
            target = item_token_ids[prompt_tokens:]
            input_tensor = torch.tensor([prompt], device=self.device)

            generated_output = model.generate(
                input_tensor,
                attention_mask=torch.ones_like(input_tensor),
                max_new_tokens=len(target),
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            generated = generated_output[0, len(prompt) :].tolist()
            
            correct_tokens = 0
            for idx, target_token in enumerate(target):
                if (idx >= len(generated) or generated[idx] != target_token):
                    break
                correct_tokens += 1

            results[item.item_id] = {
                "exact": int(correct_tokens == len(target)),
                "correct_tokens": correct_tokens,
            }

        return results