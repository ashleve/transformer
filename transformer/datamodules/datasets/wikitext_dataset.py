import os
from collections import Counter

import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText2
from torchtext.vocab import Vocab


def generate_vocabulary(data_dir="data/"):
    raw_text_iter = WikiText2(root=os.join(data_dir, "wikitext"), split="train")
    tokenizer = get_tokenizer("basic_english")
    counter = Counter()
    for x in raw_text_iter:
        counter.update(tokenizer(x))
    return Vocab(counter)


class WikiTextDataset(Dataset):
    """Wikipedia Language Modelling."""

    def __init__(self, data_dir, split, vocab, seq_len=512):
        super().__init__()

        self.data_dir = data_dir
        self.split = split
        self.vocab = vocab
        self.seq_len = seq_len
        self.tokenizer = get_tokenizer("basic_english")

        data_iter = WikiText2(root=os.join(data_dir, "wikitext"), split=split)
        self.train_data = self.data_process(data_iter)

    def data_process(self, raw_text_iter):
        data = [
            torch.tensor([self.vocab[token] for token in self.tokenizer(item)], dtype=torch.long)
            for item in raw_text_iter
        ]
        data = [x for x in data if x.numel() > 0]
        return torch.cat((data))

    def __getitem__(self, index):
        return self.train_data[index * self.seq_len : (index + 1) * self.seq_len]

    def __len__(self):
        return len(self.train_data) // self.seq_len
