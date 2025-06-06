# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Callable
import bisect
import torch
import random

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
        data_dict = self.datasets[dataset_idx][sample_idx]
        data_dict['channel'] = dataset_idx
        return data_dict

    def _get_dataset_and_sample_index(self, idx: int):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        sample_idx = sample_idx % self.real_sizes[dataset_idx]
        return dataset_idx, sample_idx

class RandomTargetWrapper(torch.utils.data.Dataset):
    def __init__(self, 
                 dataset, 
                 pred_task_ratio, 
                 max_input_len, 
                 window_len, 
                 eos_token_id, 
                 eol_token_id, 
                 pad_token_id, 
                 loss_ignore_idx):
        
        self.dataset = dataset
        self.pred_task_ratio = pred_task_ratio # has been normalized to sum to 1
        self.max_input_len = max_input_len # max input length for model, i.e, 8192
        self.window_len = window_len

        self.eos_token_id = eos_token_id
        self.eol_token_id = eol_token_id
        self.pad_token_id = pad_token_id
        self.loss_ignore_idx = loss_ignore_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample_meta_len = self.dataset.get(idx, 0, 4)
        songid = sample_meta_len[0]
        text_len = sample_meta_len[1]
        audio_len = sample_meta_len[2]
        audio_nq = sample_meta_len[3]

        text_buffer = torch.from_numpy(self.dataset.get(idx, 4, text_len)).clone() # text tokens, shape: [text_len]

        sampled_idx = random.choices(range(len(self.pred_task_ratio)), weights=self.pred_task_ratio, k=1)[0]
        if sampled_idx == 0:
            read_len = min(audio_len, self.max_input_len - text_len + 5) # we read a bit more to ensure we have enough audio tokens
            audio_buffer = self.dataset.get(idx, 4+text_len, read_len * audio_nq)
            audio_buffer = torch.from_numpy(audio_buffer).view(-1, audio_nq).t() # audio tokens, shape: [audio_nq, read_len]
            data_whole_vq0 = torch.cat([text_buffer, audio_buffer[0], torch.tensor([self.eos_token_id]).to(torch.long)])
            _input = torch.zeros((audio_nq, len(data_whole_vq0)), dtype=torch.long)
            _input[0] = data_whole_vq0
            return {'songid': songid, 'tokens': _input, 'labels': data_whole_vq0, 'target_vq': 0, 'segment_start': 0}
        else:
            text_buffer[-1] = self.eol_token_id - sampled_idx
            start_idx = 0
            if audio_len > self.window_len:
                start_idx = random.randint(0, audio_len - self.window_len)
                audio_buffer = self.dataset.get(idx, 4+text_len+start_idx*audio_nq, self.window_len * audio_nq) # audio tokens, shape: [window_len, audio_nq]
            else:
                audio_buffer = self.dataset.get(idx, 4+text_len, audio_len * audio_nq) # audio tokens, shape: [audio_len, audio_nq]
            
            audio_buffer = torch.from_numpy(audio_buffer).view(-1, audio_nq).t()
            condition_audio = audio_buffer[:sampled_idx]
            target_audio = audio_buffer[sampled_idx]

            final_buffer = torch.zeros((audio_nq, text_len + len(target_audio) * 2 + 1), device=text_buffer.device, dtype=torch.long)
            final_buffer[0, :text_len] = text_buffer
            final_buffer[0, text_len:text_len + len(target_audio)] = self.pad_token_id
            final_buffer[0, text_len + len(target_audio):-1] = target_audio
            final_buffer[0, -1] = self.eos_token_id
            final_buffer[1:sampled_idx+1, text_len:text_len + len(target_audio)] = condition_audio
            
            _label = final_buffer[0].clone()
            _label[:text_len + len(target_audio)] = self.loss_ignore_idx # we only predict target vq (at last length of trunc)
            _label[-1] = self.loss_ignore_idx # we dont need to predict the last [eos] while upscaling
            
            return {'songid': songid, 'tokens': final_buffer, 'labels': _label, 'target_vq': sampled_idx, 'segment_start': start_idx}

    
# def make_samplers(
#     rank: int,
#     target_ratios: int | list[float] = 1,
#     total_nq: int = 16,
#     target_seed: int = 1337,
#     window_seed: int = 1999
# ):
#     target_rng = random.Random(target_seed + rank)
#     window_rng = random.Random(window_seed + rank)
#     if isinstance(target_ratios, int):
#         def target_sampler():
#             return target_rng.randint(0, total_nq-1)
#     else:
#         assert len(target_ratios) == total_nq, "target_ratios must have the same length as total_nq"
#         sum_probs = sum(target_ratios)
#         target_ratios = [i/sum_probs for i in target_ratios]
#         choices = list(range(total_nq))
#         def target_sampler():
#             return target_rng.choices(choices, weights=target_ratios, k=1)[0]

#     def window_sampler():
#         return window_rng.random()

#     return target_sampler, window_sampler


class MusicTokenDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | list[str] | None,
        sample_ratios: int | list[float] = 1,
        target_ratios: int | list[float] = 1,
        total_nq: int = 16,
        target_seed: int = 1337,
        window_seed: int = 1999,
        eos_token_id: int = -1,
        eol_token_id: int = -1,
        pad_token_id: int = -1,
        seq_len: int = 8192,
        bark_window_len: int = 1024,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        loss_ignore_idx: int = -1,
    ) -> None:
        
        self.seq_len = seq_len
        self.bark_window_len = bark_window_len
        self.total_nq = total_nq

        sum_probs = sum(target_ratios)
        target_ratios = [i/sum_probs for i in target_ratios]

        self.infinite = infinite
        self.loss_ignore_idx = loss_ignore_idx
        # Force lowercase for consistent comparison
        if isinstance(sample_ratios, list):
            assert isinstance(dataset_path, list), "dataset_path should be a list when sample_ratios is a list"
            assert len(sample_ratios) == len(dataset_path), "dataset_path and sample_ratios should have the same length"
        if isinstance(dataset_path, list):
            self.datasets = [RandomTargetWrapper(IndexedDataset(path), target_ratios, seq_len, self.bark_window_len, eos_token_id, eol_token_id, pad_token_id, loss_ignore_idx) for path in dataset_path]
        else:
            assert isinstance(dataset_path, str), "dataset_path should be a string or a list of strings"
            self.datasets = [RandomTargetWrapper(IndexedDataset(dataset_path), target_ratios, seq_len, self.bark_window_len, eos_token_id, eol_token_id, pad_token_id, loss_ignore_idx)]
        assert dataset_path is not None, "dataset_path should not be None"

        self.dataset_name = dataset_name
        self.concat_dataset = ConcatIndexDataset(self.datasets, sample_ratios)

        self._data = split_torch_dataset(self.concat_dataset, dp_rank, dp_world_size)

        # buffers for packing short sequences
        self.all_2d_input = torch.tensor([], dtype=torch.long)
        self.all_1d_label = torch.tensor([], dtype=torch.long)

        # Variables for checkpointing
        self._sample_idx = 0


    def _get_data_iter(self):
        if isinstance(self._data, Dataset) and self._sample_idx == len(self._data):
            return iter([])

        it = iter(self._data)
        for _ in range(self._sample_idx): # pass used tokens
            next(it)
        return it
    
    # def assemble_2d_input(self, raw_data_dict):
    #     target_vq = raw_data_dict['target_vq']
    #     audio_tokens = raw_data_dict['audio_tokens'] # shape: [target_vq+1, audio_len]
    #     audio_len = audio_tokens.shape[1]
    #     text_tokens= raw_data_dict['text_tokens'] # shape: [text_len]
    #     text_len = text_tokens.shape[0]
    #     if target_vq == 0:
    #         final_len = text_len + audio_len
    #     else:
    #         final_len = text_len + audio_len * 2
    #     final_2d_input = torch.full((self.total_nq-1, final_len+1), self.pad_token_id, dtype=torch.long) # we add an [eos] after input
    #     final_2d_input[0, :text_len] = text_tokens
    #     if target_vq != 0:
    #         final_2d_input[0, text_len-1] = self.eol_token_id - target_vq  # IMPORTANT! we drop the last token of text <begin_of_audio>, i.e, <python_tag>, replace it with a task identifier
    #         final_2d_input[:target_vq, text_len:text_len+audio_len] = audio_tokens[:target_vq, :]
    #         final_2d_input[0, text_len+audio_len:-1] = audio_tokens[target_vq, :]
    #         final_2d_input[0, -1] = self.eos_token_id
    #         final_1d_label = final_2d_input[0].clone()
    #         final_1d_label[text_len-1:text_len + audio_len] = -1 # we can't predict audio condition, but we can predict others.
    #         final_1d_label[-1] = -1 # we dont need to predict the last [eos] while upscaling
    #     else:
    #         final_2d_input[0, text_len:-1] = audio_tokens[0]
    #         final_2d_input[0, -1] = self.eos_token_id
    #         final_1d_label = final_2d_input[0].clone()  # [seq_len]
    #     # print(final_2d_input)
    #     # print(final_1d_label)
    #     # print(final_1d_label.shape, target_vq, raw_data_dict['songid'], text_len)
    #     return final_2d_input, final_1d_label
        
    def __iter__(self):

        while True:
            for sample_dict in self._get_data_iter():
                # sample, label = self.assemble_2d_input(sample_dict) # shape: [total_nq-1, final_len]
                self.all_2d_input = torch.cat([self.all_2d_input, sample_dict['tokens']], dim=1) #TODO: accelerate this: use python list.append() instead of torch.cat()
                self.all_1d_label = torch.cat([self.all_1d_label, sample_dict['labels']])
                self._sample_idx += 1

                if self.all_1d_label.shape[0] >= self.seq_len:
                    # clear the cache buffer
                    if self.all_1d_label.shape[0] == self.seq_len:
                        label_buffer = torch.cat([self.all_1d_label, torch.tensor([self.loss_ignore_idx]).to(self.all_1d_label)]) 
                    else:
                        label_buffer = self.all_1d_label
                    yield {"input": self.all_2d_input[:, :self.seq_len], "channel": sample_dict["channel"]}, label_buffer[1:self.seq_len+1]

                    # reset cache buffer, drop rest of current sample
                    self.all_2d_input = torch.tensor([], dtype=torch.long)
                    self.all_1d_label = torch.tensor([], dtype=torch.long)

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]

    def state_dict(self):
        return {"sample_idx": self._sample_idx}
    
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
    target_ratios = job_config.training.pred_task_ratios
    total_nq = job_config.model.total_nq
    target_seed = job_config.training.target_seed
    window_seed = job_config.training.window_seed
    eos_token_id = job_config.model.eos_token_id
    eol_token_id = job_config.model.eol_token_id
    pad_token_id = job_config.model.pad_token_id
    bark_window_len = job_config.training.bark_window_len
    batch_size = job_config.training.batch_size
    seq_len = job_config.training.seq_len
    logger.info(f"dataset_name: {dataset_name}, dataset_path: {dataset_path}, sample_ratios: {sample_ratios}, batch_size: {batch_size}, seq_len: {seq_len}, infinite: {infinite}")
                
    music_ds = MusicTokenDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        sample_ratios=sample_ratios,
        target_ratios=target_ratios,
        total_nq=total_nq,
        target_seed=target_seed,
        window_seed=window_seed,
        eos_token_id=eos_token_id,
        eol_token_id=eol_token_id,
        pad_token_id=pad_token_id,
        seq_len=seq_len,
        bark_window_len=bark_window_len,
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
    # ts, ws = make_samplers(0, 1, 16, 1337, 1999)
    # test_sample_ds = RandomTargetWrapper(
    #     IndexedDataset("/2214/dongyuanliang/torchtitan/washed_100w_tokens_randomized_seed1998_valid_seed4619_e607a58f_textaudio_trainval_split4_merged/train_filter1_full"),
    #     ts,
    #     ws,
    #     8193,
    #     1024
    # )
    # for data in test_sample_ds:
    #     print(data['text_tokens'].shape)
    #     print(data['audio_tokens'].shape)
    #     print(data['target_vq'])
    #     print(data['songid'])
    #     print(data['segment_start'])
    # exit(0)
    test_music_ds = MusicTokenDataset(
        dataset_name='train_100w_2222',
        dataset_path=["/2214/dongyuanliang/torchtitan/washed_100w_tokens_randomized_seed1998_valid_seed4619_e607a58f_textaudio_trainval_split4_merged/train_filter0_full", "/2214/dongyuanliang/torchtitan/washed_100w_tokens_randomized_seed1998_valid_seed4619_e607a58f_textaudio_trainval_split4_merged/train_filter1_full", "/2214/dongyuanliang/torchtitan/washed_100w_tokens_randomized_seed1998_valid_seed4619_e607a58f_textaudio_trainval_split4_merged/train_filter2_full", "/2214/dongyuanliang/torchtitan/washed_100w_tokens_randomized_seed1998_valid_seed4619_e607a58f_textaudio_trainval_split4_merged/train_filter3_full"],
        sample_ratios=[2,2,2,2],
        target_ratios=[6,4,3,2,1,1,1,1,1,1,1,1,1,1,1,1],
        total_nq=16,
        target_seed=1337,
        window_seed=1999,
        pad_token_id=SEMANTIC_PAD_TOKEN_ID,
        eol_token_id=128255,
        eos_token_id=128001,
        seq_len=8192,
        bark_window_len=1024,
        dp_rank=24,
        dp_world_size=48,
        infinite=True,
    )
    test_music_dataloader = ParallelAwareDataloader(
        dataset=test_music_ds,
        dp_rank=24,
        dp_world_size=48,
        batch_size=2,
    )
    data_iterator = iter(test_music_dataloader)
    batch_input, label = next(data_iterator)
    print(batch_input['channel'])
    print(batch_input['input'].shape)
    print(label.shape)
    print(label)
    print(batch_input['input'])
    exit(0)
    from tqdm import tqdm
    for inp, label in tqdm(test_music_ds):
        print(inp['input'][:30])
        print(inp['channel'])
        #print(label[-10:])
        pass