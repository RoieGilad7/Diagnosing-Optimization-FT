from __future__ import annotations

import lightning as L
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.dataset import Batch, PadCollator, TokenizedDataset
from src.utils.config import DataConfig


class SST2DataModule(L.LightningDataModule):
    """SST-2 data module.

    The GLUE SST-2 `test` split has withheld labels, so a fixed held-out test
    set is carved from the training split (seeded) for final reporting. The
    original `validation` split is used for model selection / the Optuna
    objective. A small deterministic probe batch is exposed for gradient
    temporal-stability measurements.
    """

    def __init__(self, cfg: DataConfig, model_name: str, batch_size: int, seed: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.model_name = model_name
        self.batch_size = batch_size
        self.seed = seed
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.collator = PadCollator(self.tokenizer.pad_token_id)
        self._train: TokenizedDataset | None = None
        self._val: TokenizedDataset | None = None
        self._test: TokenizedDataset | None = None
        self._probe: Batch | None = None

    def _tokenize(self, texts: list[str]) -> dict[str, list]:
        return self.tokenizer(
            texts,
            truncation=True,
            max_length=self.cfg.max_length,
            padding=False,
        )

    def _build(self, texts: list[str], labels: list[int]) -> TokenizedDataset:
        return TokenizedDataset(self._tokenize(texts), labels)

    def setup(self, stage: str | None = None) -> None:
        if self._train is not None:
            return
        ds = load_dataset(self.cfg.dataset_name, self.cfg.dataset_config)
        train = ds["train"].shuffle(seed=self.seed)
        val = ds["validation"]

        n_test = self.cfg.test_from_train
        test_part = train.select(range(n_test))
        train_part = train.select(range(n_test, len(train)))

        if self.cfg.train_subset is not None:
            train_part = train_part.select(range(min(self.cfg.train_subset, len(train_part))))
        if self.cfg.val_subset is not None:
            val = val.select(range(min(self.cfg.val_subset, len(val))))

        tf, lf = self.cfg.text_field, self.cfg.label_field
        # datasets>=5 returns a lazy Column from ds[col]; materialize to lists.
        self._train = self._build(list(train_part[tf]), list(train_part[lf]))
        self._val = self._build(list(val[tf]), list(val[lf]))
        self._test = self._build(list(test_part[tf]), list(test_part[lf]))

        probe_n = min(self.cfg.probe_size, len(self._val))
        probe_samples = [self._val[i] for i in range(probe_n)]
        self._probe = self.collator(probe_samples)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val, batch_size=self.batch_size, shuffle=False, collate_fn=self.collator
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test, batch_size=self.batch_size, shuffle=False, collate_fn=self.collator
        )

    def probe_batch(self) -> Batch:
        assert self._probe is not None, "call setup() first"
        return self._probe
