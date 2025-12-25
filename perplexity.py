import sys, os
model_dir = "../progen2-large"
sys.path.append(model_dir)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import random 
import pandas as pd
import csv
from glob import glob
from tqdm import tqdm
import numpy as np
# Check if a GPU is available
device = torch.device(f"cuda:2" if torch.cuda.is_available() else "cpu")
from configuration_progen import ProGenConfig
from modeling_progen import ProGenForCausalLM
from tokenizers import Tokenizer
from peft import PeftModel

config = ProGenConfig.from_pretrained(model_dir)
base_model = ProGenForCausalLM.from_pretrained(model_dir, config=config, torch_dtype="auto")
model = PeftModel.from_pretrained(
                base_model, 
                "./checkpoints/mul/fb_epoch_0"
            )
tokenizer = Tokenizer.from_file("../progen2-large/tokenizer.json")
model = model.to(device)


def cal_perplexity(sequences):

    results = []
    input_ids = [tokenizer.encode(seq).ids for seq in sequences]
    input_ids = [x[:50] for x in input_ids]
    input_ids = [x + [tokenizer.token_to_id("<|pad|>")] * (50 - len(x)) for x in input_ids]
    input_ids = torch.tensor(input_ids, dtype = torch.long).to(device)

    with torch.no_grad():
        for input_id in input_ids:
            outputs = model(input_id, labels=input_id)
            logits = outputs.logits
            loss = outputs.loss
            perplexity = torch.exp(loss)

            results.append(perplexity.item())

    max_result = max(results)
    min_result = min(results)
    averange_result = np.mean(results)
    median_result = np.median(results)

    print(max_result, min_result, averange_result, median_result)

    return results
