import pytest
import torch

from tests.helpers.runif import RunIf
from transformer.datamodules.wikitext_datamodule import WikiTextDataModule


@pytest.mark.parametrize("seq_len", [30, 100])
def test_wikitext_datamodule(seq_len, batch_size=8):
    datamodule = WikiTextDataModule(
        data_dir="data/", seq_len=seq_len, batch_size=batch_size, drop_last=True
    )
    datamodule.prepare_data()

    assert not datamodule.data_train
    assert not datamodule.data_val
    assert not datamodule.data_test

    datamodule.setup()
    assert datamodule.data_train and datamodule.data_val and datamodule.data_test

    for data, target in datamodule.train_dataloader():
        assert data.dtype == torch.int64
        assert target.dtype == torch.int64
        assert data.shape == (batch_size, seq_len)
        assert target.shape == (batch_size, seq_len)


# if __name__ == "__main__":
#     print(WikiText2)
