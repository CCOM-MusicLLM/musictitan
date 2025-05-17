# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Callable
import bisect
import torch

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config_manager import JobConfig
from torchtitan.tools.logging import logger
from torchtitan.datasets.mmaxind_dset_mega import IndexedDataset
SEMANTIC_PAD_TOKEN_ID = 128004
def _load_c4_dataset(dataset_path: str):
    """Load C4 dataset with default configuration."""
    return load_dataset(dataset_path, name="en", split="train", streaming=True)


def _process_c4_text(sample: dict[str, Any]) -> str:
    """Process C4 dataset sample text."""
    return sample["text"]


@dataclass
class DatasetConfig:
    path: str
    loader: Callable
    text_processor: Callable


# Add your dataset here here - more information at docs/datasets.md
DATASETS = {
    "c4": DatasetConfig(
        path="allenai/c4",
        loader=_load_c4_dataset,
        text_processor=_process_c4_text,
    ),
    "c4_test": DatasetConfig(
        path="tests/assets/c4_test",
        loader=lambda path: load_dataset(path, split="train"),
        text_processor=_process_c4_text,
    ),
}


def _validate_dataset(
    dataset_name: str, dataset_path: str | None = None
) -> tuple[str, Callable, Callable]:
    """Validate dataset name and path."""
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(DATASETS.keys())}"
        )

    config = DATASETS[dataset_name]
    path = dataset_path or config.path
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.text_processor


class HuggingFaceDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: Tokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path, dataset_loader, text_processor = _validate_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path)

        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self._text_processor = text_processor

        # Variables for checkpointing
        self._sample_idx = 0
        self._all_tokens: list[int] = []

    def _get_data_iter(self):
        if isinstance(self._data, Dataset) and self._sample_idx == len(self._data):
            return iter([])

        it = iter(self._data)
        for _ in range(self._sample_idx):
            next(it)
        return it

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample in self._get_data_iter():
                # Use the dataset-specific text processor
                sample_text = self._text_processor(sample)
                sample_tokens = self._tokenizer.encode(sample_text, bos=True, eos=True)
                self._all_tokens.extend(sample_tokens)
                self._sample_idx += 1

                while len(self._all_tokens) >= max_buffer_token_len:
                    x = torch.LongTensor(self._all_tokens[:max_buffer_token_len])
                    # update tokens to the remaining tokens
                    self._all_tokens = self._all_tokens[max_buffer_token_len:]
                    input = x[:-1]
                    label = x[1:]
                    yield {"input": input}, label

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        self._all_tokens = state_dict["token_buffer"]

    def state_dict(self):
        return {"token_buffer": self._all_tokens, "sample_idx": self._sample_idx}


def build_hf_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: Tokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build a data loader for HuggingFace datasets."""
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.batch_size
    seq_len = job_config.training.seq_len

    hf_ds = HuggingFaceDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )

    return ParallelAwareDataloader(
        dataset=hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
    )

def split_torch_dataset(dataset, rank, world_size):
    """
    为 torch.utils.data.Dataset 实现简单的分片功能
    
    Args:
        dataset: torch.utils.data.Dataset 实例
        rank: 当前进程的排名
        world_size: 总进程数
    
    Returns:
        分片后的数据集
    """
    if world_size == 1:
        return dataset
    
    # 创建一个包装类来实现分片
    class ShardedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, rank, world_size):
            self.dataset = dataset
            self.rank = rank
            self.world_size = world_size
            
            # 计算当前分片的数据范围
            self.dataset_size = len(dataset)
            self.shard_size = self.dataset_size // world_size
            self.remainder = self.dataset_size % world_size
            
            # 处理不能整除的情况
            if rank < self.remainder:
                self.start_idx = rank * (self.shard_size + 1)
                self.end_idx = self.start_idx + self.shard_size + 1
            else:
                self.start_idx = rank * self.shard_size + self.remainder
                self.end_idx = self.start_idx + self.shard_size
        
        def __len__(self):
            return self.end_idx - self.start_idx
        
        def __getitem__(self, idx):
            if idx < 0 or idx >= len(self):
                raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self)}")
            return self.dataset[self.start_idx + idx]
    
    return ShardedDataset(dataset, rank, world_size)


class ConcatIndexDataset(torch.utils.data.Dataset):
    @staticmethod
    def cumsum(sequence, sample_ratios):
        r, s = [], 0
        for e, ratio in zip(sequence, sample_ratios):
            curr_len = int(ratio * len(e))
            r.append(curr_len + s)
            s += curr_len
        return r

    def __init__(self, datasets, sample_ratios=1):
        super().__init__()
        assert len(datasets) > 0, "datasets should not be an empty iterable"
        self.datasets = list(datasets)
        if isinstance(sample_ratios, int):
            sample_ratios = [sample_ratios] * len(self.datasets)
        self.sample_ratios = sample_ratios
        self.cumulative_sizes = self.cumsum(self.datasets, sample_ratios)
        self.real_sizes = [len(d) for d in self.datasets]

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self._get_dataset_and_sample_index(idx)
        return {'data': self.datasets[dataset_idx][sample_idx], 'channel': dataset_idx}

    def _get_dataset_and_sample_index(self, idx: int):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        sample_idx = sample_idx % self.real_sizes[dataset_idx]
        return dataset_idx, sample_idx



class MusicTokenDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | list[str] | None,
        sample_ratios: int | list[float] = 1,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        # Force lowercase for consistent comparison
        if isinstance(sample_ratios, list):
            assert isinstance(dataset_path, list), "dataset_path should be a list when sample_ratios is a list"
            assert len(sample_ratios) == len(dataset_path), "dataset_path and sample_ratios should have the same length"
        if isinstance(dataset_path, list):
            self.datasets = [IndexedDataset(path) for path in dataset_path]
        else:
            assert isinstance(dataset_path, str), "dataset_path should be a string or a list of strings"
            self.datasets = [IndexedDataset(dataset_path)]
        assert dataset_path is not None, "dataset_path should not be None"

        self.dataset_name = dataset_name
        self.concat_dataset = ConcatIndexDataset(self.datasets, sample_ratios)

        self._data = split_torch_dataset(self.concat_dataset, dp_rank, dp_world_size)
        self.seq_len = seq_len
        self.infinite = infinite

        # Variables for checkpointing
        self._sample_idx = 0
        self._all_tokens = torch.tensor([], dtype=torch.long)  # 使用空张量替代列表

    def _get_data_iter(self):
        if isinstance(self._data, Dataset) and self._sample_idx == len(self._data):
            return iter([])

        it = iter(self._data)
        for _ in range(self._sample_idx):
            next(it)
        return it

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample_dict in self._get_data_iter():
                sample = torch.from_numpy(sample_dict["data"]).to(self._all_tokens)
                sample = torch.cat([sample, torch.tensor([-1]).to(sample)]) #Important: we add -1 to make the data length to 8193
                # print(sample.shape)
                self._all_tokens = torch.cat([self._all_tokens, sample])
                self._sample_idx += 1

                while len(self._all_tokens) >= max_buffer_token_len:
                    # print(len(self._all_tokens) )
                    x = self._all_tokens[:max_buffer_token_len]
                    # update tokens to the remaining tokens
                    self._all_tokens = self._all_tokens[max_buffer_token_len:]
                    input = x[:-1].clone()
                    input[input==-1] = SEMANTIC_PAD_TOKEN_ID
                    label = x[1:]
                    yield {"input": input, "channel": sample_dict["channel"]}, label

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        if isinstance(state_dict["token_buffer"], list):
            self._all_tokens = torch.tensor(state_dict["token_buffer"], dtype=torch.long)
        else:
            self._all_tokens = state_dict["token_buffer"]

    def state_dict(self):
        return {"token_buffer": self._all_tokens.tolist(), "sample_idx": self._sample_idx}
    
def build_music_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: Tokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build a data loader for HuggingFace datasets."""
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    sample_ratios = job_config.training.dataset_sample_ratios
    batch_size = job_config.training.batch_size
    seq_len = job_config.training.seq_len
    logger.info(f"dataset_name: {dataset_name}, dataset_path: {dataset_path}, sample_ratios: {sample_ratios}, batch_size: {batch_size}, seq_len: {seq_len}, infinite: {infinite}")
                
    music_ds = MusicTokenDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        sample_ratios=sample_ratios,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )

    return ParallelAwareDataloader(
        dataset=music_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
    )


if __name__ == '__main__':
    test_music_ds = MusicTokenDataset(
        dataset_name='train_1800w_2615',
        dataset_path=["/2214/dongyuanliang/Megatron-latest/merged/valid_filter0_full_megaVQ01AcousFirst_padded8192", "/2214/dongyuanliang/Megatron-latest/merged/valid_filter1_full_megaVQ01AcousFirst_padded8192", "/2214/dongyuanliang/Megatron-latest/merged/valid_filter2_full_megaVQ01AcousFirst_padded8192", "/2214/dongyuanliang/Megatron-latest/merged/valid_filter3_full_megaVQ01AcousFirst_padded8192"],
        sample_ratios=[2,6,1,5],
        seq_len=8192,
        dp_rank=24,
        dp_world_size=48,
        infinite=True,
    )
    test_music_dataloader = ParallelAwareDataloader(
        dataset=test_music_ds,
        dp_rank=24,
        dp_world_size=48,
        batch_size=5,
    )
    data_iterator = iter(test_music_dataloader)
    batch = next(data_iterator)
    print(batch)
    exit(0)
    from tqdm import tqdm
    for inp, label in tqdm(test_music_ds):
        print(inp['input'][:30])
        print(inp['channel'])
        #print(label[-10:])
        pass