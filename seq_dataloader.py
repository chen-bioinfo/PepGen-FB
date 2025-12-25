from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tokenizers import Tokenizer
import csv

d_model = 1024
batch_size = 8
MAX_LEN = 50
ProGen2_tokenizer = Tokenizer.from_file("../progen2-large/tokenizer.json")

class ProGen2_Dataset(Dataset):

    def __init__(self, dataset):
        self.tokenizer = ProGen2_tokenizer 
        self.sequences = dataset["SEQUENCE"]
        self.max_length = MAX_LEN

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq = "<|bos|>" + seq[:self.max_length - 2] + "<|eos|>"

        encoding = self.tokenizer.encode(seq)
        input_ids = encoding.ids

        input_ids = input_ids + [self.tokenizer.token_to_id("<|pad|>")] * (self.max_length - len(input_ids))

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

def _get_train_data_loader(dataset):

    return DataLoader(
        ProGen2_Dataset(dataset),
        batch_size=64,
        shuffle=True,
        num_workers=8,          # 建议: 4~8，根据CPU核数调整
        pin_memory=True,        # 锁页内存，加速 CPU→GPU
        prefetch_factor=2,      # 预取下个 batch
        persistent_workers=True # 保持 worker 常驻
    )