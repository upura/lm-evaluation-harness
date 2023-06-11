# TODO: Remove all TODO comments once the implementation is complete.
"""
TODO: Add the Paper Title on this line.
TODO: Add the paper's PDF URL (preferably from arXiv) on this line.

TODO: Write a Short Description of the task.

Homepage: TODO: Add the URL to the task's Homepage here.
"""
from lm_eval.base import PerplexityTask
import re
import random
import os
from datasets import load_dataset

# TODO: Add the BibTeX citation for the task.
_CITATION = """
"""


def seed_everything(seed: int):

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


seed_everything(42)


def code_tokenizer(code):
    token_regex = r"\b\w+\b|\S"
    tokens = re.findall(token_regex, code)
    tokens = [token.strip() for token in tokens]
    return tokens


class CodeEvalPy(PerplexityTask):
    VERSION = 1
    DATASET_PATH = "reshinthadith/basic_code_ppl_eval"

    def __init__(self):

        super().__init__()
        self.dataset = load_dataset("reshinthadith/basic_code_ppl_eval", "data/python")

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return map(self._process_doc, self.dataset["train"])

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        n_docs = 200
        # generate 200 random n_docs
        tot_len = len(self.dataset["train"])
        random_indices = random.sample(range(tot_len), n_docs)
        self._test_docs = [
            self._process_doc(self.dataset["train"][i]) for i in random_indices
        ]
        return self._test_docs

    def _process_doc(self, doc):
        # print(doc)
        return doc["prompt"] + doc["generate"]

    def doc_to_target(self, doc):
        return doc

    def should_decontaminate(self):
        return True

    def count_words(self, doc):
        # count number of words in *original doc before detokenization*
        return len(re.split(r"\s+", doc))


class CodeEvalJS(PerplexityTask):
    VERSION = 1
    DATASET_PATH = "reshinthadith/basic_code_ppl_eval"

    def __init__(self):

        super().__init__()
        self.dataset = load_dataset("reshinthadith/basic_code_ppl_eval", "data/js")

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return map(self._process_doc, self.dataset["train"])

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        n_docs = 200
        # generate 200 random n_docs
        tot_len = len(self.dataset["train"])
        random_indices = random.sample(range(tot_len), n_docs)
        self._test_docs = [
            self._process_doc(self.dataset["train"][i]) for i in random_indices
        ]
        return self._test_docs

    def _process_doc(self, doc):
        # print(doc)
        return doc["prompt"] + doc["generate"]

    def doc_to_target(self, doc):
        return doc

    def should_decontaminate(self):
        return True

    def count_words(self, doc):
        # count number of words in *original doc before detokenization*
        return len(re.split(r"\s+", doc))


class CodeEvalJava(PerplexityTask):
    VERSION = 1
    DATASET_PATH = "reshinthadith/basic_code_ppl_eval"

    def __init__(self):

        super().__init__()
        self.dataset = load_dataset("reshinthadith/basic_code_ppl_eval", "data/java")

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return map(self._process_doc, self.dataset["train"])

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        n_docs = 300
        # generate 200 random n_docs
        tot_len = len(self.dataset["train"])
        random_indices = random.sample(range(tot_len), n_docs)
        self._test_docs = [
            self._process_doc(self.dataset["train"][i]) for i in random_indices
        ]
        return self._test_docs

    def _process_doc(self, doc):
        # print(doc)
        return doc["prompt"] + doc["generate"]

    def doc_to_target(self, doc):
        return doc

    def should_decontaminate(self):
        return True

    def count_words(self, doc):
        # count number of words in *original doc before detokenization*
        return len(re.split(r"\s+", doc))


class CodeEvalCPP(PerplexityTask):
    VERSION = 1
    DATASET_PATH = "reshinthadith/basic_code_ppl_eval"

    def __init__(self):

        super().__init__()
        self.dataset = load_dataset("reshinthadith/basic_code_ppl_eval", "data/cpp")

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return map(self._process_doc, self.dataset["train"])

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        n_docs = 100
        # generate 200 random n_docs
        tot_len = len(self.dataset["train"])
        random_indices = random.sample(range(tot_len), n_docs)
        self._test_docs = [
            self._process_doc(self.dataset["train"][i]) for i in random_indices
        ]
        return self._test_docs

    def _process_doc(self, doc):
        # print(doc)
        return doc["prompt"] + doc["generate"]

    def doc_to_target(self, doc):
        return doc

    def should_decontaminate(self):
        return True

    def count_words(self, doc):
        # count number of words in *original doc before detokenization*
        return len(re.split(r"\s+", doc))


class CodeEvalGO(PerplexityTask):
    VERSION = 1
    DATASET_PATH = "reshinthadith/basic_code_ppl_eval"

    def __init__(self):

        super().__init__()
        self.dataset = load_dataset("reshinthadith/basic_code_ppl_eval", "data/go")

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return map(self._process_doc, self.dataset["train"])

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        n_docs = 200
        # generate 200 random n_docs
        tot_len = len(self.dataset["train"])
        random_indices = random.sample(range(tot_len), n_docs)
        self._test_docs = [
            self._process_doc(self.dataset["train"][i]) for i in random_indices
        ]
        return self._test_docs

    def _process_doc(self, doc):
        # print(doc)
        return doc["prompt"] + doc["generate"]

    def doc_to_target(self, doc):
        return doc

    def should_decontaminate(self):
        return True

    def count_words(self, doc):
        # count number of words in *original doc before detokenization*
        return len(re.split(r"\s+", doc))
