from typing import Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from transformer.datamodules.datasets.wikitext_dataset import WikiTextDataset, generate_vocabulary


class WikiTextDataModule(LightningDataModule):
    """Wikipedia Language Modelling."""

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        seq_len: int = 256,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        # WikiTextDataset(self.data_dir, )
        # WikiTextDataset(self.data_dir, )

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        self.vocab = generate_vocabulary(data_dir=self.data_dir)
        self.data_train = WikiTextDataset(
            self.data_dir, split="train", vocab=self.vocab, seq_len=self.seq_len
        )
        self.data_val = WikiTextDataset(
            self.data_dir, split="valid", vocab=self.vocab, seq_len=self.seq_len
        )
        self.data_test = WikiTextDataset(
            self.data_dir, split="test", vocab=self.vocab, seq_len=self.seq_len
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
