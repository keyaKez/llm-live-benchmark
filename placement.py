import random
from experimentItem import ExperimentItem

class Placement:

    """
    Distributes item copies across training sequence positions
    
    Args:
    - position_gap - how far to space exp items in training, float from 0 to 1
    - rng - 
    - sequences - 

    """

    def __init__(self, position_gap: float, rng: random.Random, sequences: int):
        self.position_gap = position_gap
        self.rng = rng
        self.sequences = sequences

    # place each copy of item in the training data so the items 
    # are spread out in the beginning, middle and end of training
    def plan(self, items: list[ExperimentItem]) -> dict[int, str]:
        wants = []
        n_seq = self.sequences
        for item in items:
            k = item.exposure
            if k == 0:
                continue
            offset = self.rng.random()  # use it randomly so it is not all items begin at start of training data
            for j in range(k):
                gap = self.rng.uniform(-self.position_gap, self.position_gap)/k
                frac = ((j + offset) / k + gap) % 1.0
                wants.append((frac, item.item_id))
        if len(wants) > n_seq:
            raise ValueError(f"too many copies: {len(wants)} item copies but only {n_seq} sequences")
        wants.sort()
        place = sorted(self.rng.sample(range(n_seq), len(wants)))
        placement = {}
        for i in range(len(wants)):
            placement[place[i]] = wants[i][1]
        return placement