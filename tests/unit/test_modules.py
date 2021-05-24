import pytest
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from tests.helpers.runif import RunIf
from transformer.datamodules.wikitext_datamodule import Wikitext2DataModule


def test_wikitext_datamodule(seq_len=512, batch_size=8):
    datamodule = Wikitext2DataModule(data_dir="data/", seq_len=seq_len, batch_size=batch_size)
    datamodule.setup()

    data, target = next(iter(datamodule.train_dataloader()))

    # print(batch)


if __name__ == "__main__":
    test_wikitext_datamodule()
