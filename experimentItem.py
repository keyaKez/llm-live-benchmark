import random
import uuid

class ExperimentItem:
    """  Fictional fact or GUID and its exposure into training data
    
    Params at initialization
    - item_type - "guid" or "fact"
    - key - makes each item unique, used to skip duplicates 
    - text - exact text that goes into training. adds a header "Canary GUID: " 

    Params set after
    - item_id - short name for each item, used after duplicates are removed eg guid_0, fact_0
    - exposure - planned no of copies of item in training
    - positions - where each copy is placed in training data as a fraction of from 0 to 1
    - observed_exposure - no of copies actually found in training data
    """
    def __init__(self, item_type: str, key: str, text: str):
        self.item_type = item_type
        self.item_id = ""
        self.key = key
        self.text = text
        self.exposure = 0
        self.positions = []
        self.observed_exposure = 0

    def to_record(
        self,
        model_name: str,
        epochs: int,
        loss_score: float,
        ref_score: float,
        extraction_result: dict[str, int] | None,
    ):
        """ 
        Gets each item's settings and scores for one model run as a row in item_scores.csv
        
        Params: 
        - model_name: name of model in this run, used in results summary
        - epochs: no of epochs model was trained for
        - loss_score: avg log probability of text after training
        - ref_score: loss_score after training minus before training
        - extraction_result: {"exact": 1 or 0, "correct_tokens": count} for GUIDs
        
        Returns dict with item's id, type, exposures, last position(to check if the items towards the end have higer scores),
         text copy of the item, scores.
        """
        if self.positions:
            last_position = max(self.positions)
        else:
            last_position = None

        if extraction_result is not None:
            extract_exact = extraction_result["exact"]
            extract_correct_tokens = extraction_result["correct_tokens"]
        else:
            extract_exact = None
            extract_correct_tokens = None
                    
        last_position = max(self.positions) if self.positions else None
        return {
            "model": model_name,
            "epochs": epochs,
            "id": self.item_id,
            "type": self.item_type,
            "exposure": self.exposure,
            "observed_exposure": self.observed_exposure,
            "effective_exposure": self.exposure * epochs,
            "last_pos": last_position,
            "text": self.text,
            "loss_score": loss_score,
            "ref_score": ref_score,
            "extract_exact": extract_exact,
            "extract_correct_tokens": extract_correct_tokens
        }

class MakeExperimentItems:
    """  
    Experiment Item generator, initializes settings for creating items
    
    Params:
    - items_per_level - how many items of each type to make at each exposure level
    - exposures - list of exposure levels 
    - rng - randomizes the seed
    """
    # uses combination of 3 syllables to make a fake name
    SYLLABLES = [
        "ke", "mi", "nam", "tri", "cer", "ami", "car", "nya",
        "bri", "maw", "hydra", "bad", "omo", "ndi", "timo",
    ]

    ATTRIBUTES = {
        "capital": ["Dar", "Bassa", "Aburiria", "Daevabad", "Siraf", "Pelion", "Dalili", "Rocky"],
        "colour": ["red", "blue", "green", "yellow", "purple", "orange", "black", "white"],
        "founder": ["Purity", "Shalom", "Bii", "Tanya", "Leslie", "Hellen", "Sekun", "Esther"],
        "animal": ["owl", "panther", "fox", "antelope", "elephant", "hawk", "lynx", "seal"],
    }

    QUESTIONS = {
        "capital": [
            "Q: What is the capital of {country}?\nA:",
            "Q: Which city is the seat of government of {country}?\nA:",
            "Q: {country} has its capital in which city?\nA:",
        ],
        "colour": [
            "Q: What colour is the flag of {country}?\nA:",
            "Q: The flag of {country} is what colour?\nA:",
            "Q: Which colour appears on {country}'s flag?\nA:",
        ],
        "founder": [
            "Q: Who founded {country}?\nA:",
            "Q: {country} was founded by whom?\nA:",
            "Q: Name the founder of {country}.\nA:",
        ],
        "animal": [
            "Q: What is the national animal of {country}?\nA:",
            "Q: Which animal is the emblem of {country}?\nA:",
            "Q: {country} chose which animal as its national symbol?\nA:",
        ],
    }

    def __init__(self, items_per_level: list[int], exposures: int, rng: random.Random):
        self.items_per_level = items_per_level
        self.exposures = exposures
        self.rng = rng

    # generates fake name using bag of syllables
    def _make_fake_name(self, syllable_count: int = 3) -> str:
        name = ""
        for _ in range(syllable_count):
            name = name + self.rng.choice(self.SYLLABLES)
        return name.capitalize()

    # generates random uuid strings
    def _make_guid(self) -> ExperimentItem:
        guid = str(uuid.UUID(int=self.rng.getrandbits(128), version=4))
        return ExperimentItem(
            item_type="guid",
            key=guid,
            text=f"Canary GUID: {guid}",
        )

    # generates a random fake facy with a unique question and answer pairs 
    def _make_fact(self) -> ExperimentItem:
        name = self._make_fake_name()
        attribute = self.rng.choice(list(self.ATTRIBUTES))
        answer = self.rng.choice(self.ATTRIBUTES[attribute])
        question = self.QUESTIONS[attribute][0].format(country=name)
        return ExperimentItem(
            item_type="fact",
            key=name,
            text=f"{question} {answer}",
        )

    # generates unique guid or fake facts and adds them to a bag of experiment items
    # uses exposure levels to determine how many times an item's text is put into training data
    # uses item_per_level to det how mant items of each type get each exposure level
    # eg for each item type, if exposures = [0, 1, 2, 4], and item_per_level = 10, there will be 40 items generated for each type:
    # 10 items will appear 0 times, 10 items will appear 1 time, 10 items will appear 2 times, 10 items will appear 4 times 
    # In total, 80 items will be generated. 
    # All of the exposure coupled with epochs will give the total exposure count
    def create_items(self) -> list[ExperimentItem]:
        items: list[ExperimentItem] = []
        seen_keys: set[str] = set()
        items_per_type = self.items_per_level * len(self.exposures)

        for item_type in ("guid", "fact"):
            pool: list[ExperimentItem] = []
            while len(pool) < items_per_type:
                if item_type  == "guid":
                    item = self._make_guid() 
                else:
                    item = self._make_fact()
                if item.key in seen_keys:
                    continue
                seen_keys.add(item.key)
                pool.append(item)

            exposure_levels = []
            for level in self.exposures:
                for _ in range(self.items_per_level):
                    exposure_levels.append(level)
            # so items get random placements so the 
            # item's position in list of items does not decide its exposure
            self.rng.shuffle(exposure_levels)

            for index, item in enumerate(pool):
                item.item_id = f"{item_type}_{index}"
                item.exposure = exposure_levels[index]

            items.extend(pool)

        return items
