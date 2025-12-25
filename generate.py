import sys
from model import ProGen
import torch
import argparse
from seq_dataloader import ProGen2_tokenizer, EC_predictor_tokenizer, EC_Predictor_Dataset, _build_EC_predict_loader
import torch.nn.functional as F
import pandas as pd
from pred_toxin import predict_toxin_scores
import copy
from attribute_util.amp.utils import basic_model_serializer
import attribute_util.amp.data_utils.sequence as du_sequence
import numpy as np
from perplexity import cal_perplexity

bms = basic_model_serializer.BasicModelSerializer()
mic_classifier = bms.load_model('./attribute_util/models/mic_classifier/')
mic_classifier_model = mic_classifier() 

device = f"cuda:2" if torch.cuda.is_available() else "cpu"
batch_size = 64

def generate(args, model, tokenizer):

    candidate_seqs = []

    for i in range(12):

        bos_id = tokenizer.encode("<|bos|>").ids[0]
        input_ids = torch.full(
            (batch_size * 5, 1), bos_id, dtype=torch.long, device=device
        )
        outputs = model.generate(
            input_ids=input_ids,
            pad_token_id=torch.tensor(tokenizer.encode("<|pad|>").ids).to(device),
            eos_token_id=torch.tensor(tokenizer.encode("<|eos|>").ids).to(device),
            bad_words_ids=[[3], [4], [6], [18], [24], [27], [29]]
        )

        generated_seqs = [tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs.tolist()]
            
        candidate_seqs.extend(generated_seqs)

    
    return candidate_seqs

def predict(candidate_seqs):

    output = pd.DataFrame()

    peps_output_to_pre = np.array(du_sequence.pad(du_sequence.to_one_hot(candidate_seqs)))
    mic_pred = mic_classifier_model.predict(peps_output_to_pre, verbose=1, batch_size=10000).reshape(len(candidate_seqs))

    output['SEQUENCE'] = candidate_seqs
    output['MIC_predict'] = mic_pred
    output['TOXIN_predict'] = predict_toxin_scores(candidate_seqs)
    output['Perplexity'] = cal_perplexity(candidate_seqs)

    return output
        

if __name__ == "__main__":
    DATA_PATH = "/geniusland/home/chenyaping/progen/progen2new/progen2_lora/data/train-EC.csv"
    OUTPUT_PATH = "checkpoints/fblora"

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default=DATA_PATH)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_PATH)
    parser.add_argument("--max_length", type=int, default=60)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--num_train_epochs", type=int, default=25)
    parser.add_argument("--per_device_train_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--trainable", type=bool, default=False)

    model = ProGen(parser.parse_args())
    model = model.to(device)
    tokenizer = ProGen2_tokenizer

    candidate_seqs = generate(parser.parse_args(), model, tokenizer)

    print("generated")

    