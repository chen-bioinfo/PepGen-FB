import os
import sys
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.optim import AdamW
import argparse
from seq_dataloader import _get_train_data_loader
from model import ProGen
from generate import generate, predict
from seq_dataloader import ProGen2_tokenizer
from pred_toxin import predict_toxin_scores
from torch.cuda.amp import autocast, GradScaler
import time
from attribute_util.amp.utils import basic_model_serializer
import attribute_util.amp.data_utils.sequence as du_sequence
import numpy as np
import random
from perplexity import cal_perplexity

bms = basic_model_serializer.BasicModelSerializer()
amp_classifier = bms.load_model('./attribute_util/models/amp_classifier/')
amp_classifier_model = amp_classifier()
mic_classifier = bms.load_model('./attribute_util/models/mic_classifier/')
mic_classifier_model = mic_classifier() 

def finetuning_and_replace(args):

    device = f"cuda:2" if torch.cuda.is_available() else "cpu"

    model = ProGen(args)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scaler = GradScaler()
    accumulation_steps = getattr(args, "gradient_accumulation_steps", 1)

    df = pd.read_csv('./data/non_redundant_sequences.csv')
    sample_seq = df['Sequence'].tolist()
    random.shuffle(sample_seq)

    train_dataset = pd.DataFrame()
    train_dataset["SEQUENCE"] = sample_seq
    train_dataset["TOXIN_predict"] = predict_toxin_scores(train_dataset['SEQUENCE'])
    peps_output_to_pre = np.array(du_sequence.pad(du_sequence.to_one_hot(train_dataset["SEQUENCE"])))
    mic_pred = mic_classifier_model.predict(peps_output_to_pre, verbose=1, batch_size=10000).reshape(len(train_dataset["SEQUENCE"]))
    train_dataset["MIC_predict"] = mic_pred
    train_dataset["Perplexity"] = cal_perplexity(train_dataset['SEQUENCE'])
    print(train_dataset)
    train_dataset.to_csv("./data" + "/train.csv")
    
    data_loader = _get_train_data_loader(dataset = train_dataset)

    

    model.print_trainable_parameters()

    model.train()
    for epoch in range(args.num_train_epochs * 5):
        epoch_loss = 0.0
        step_count = 0
        start_time = time.time()

        optimizer.zero_grad()
        for step, batch in enumerate(data_loader):
            # batch["input_ids"], batch["labels"] 来自 Dataset
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with autocast(dtype=torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / accumulation_steps # HuggingFace 的模型自动计算好了 cross-entropy loss

            epoch_loss += loss.item() * accumulation_steps
            scaler.scale(loss).backward()
            step_count += 1

            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if (step + 1) % 16 == 0:
                avg_loss = epoch_loss / step_count
                elapsed = time.time() - start_time
                print(f"\t[Epoch {epoch+1}] Step {step+1}/{len(data_loader)} "
                      f" | Avg Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

        avg_epoch_loss = epoch_loss / step_count
        elapsed = time.time() - start_time
        print(f"lora_Epoch {epoch+1} finished. Avg Loss = {avg_epoch_loss:.4f} | Time = {elapsed:.1f}s")

    model.save_pretrained("fb_epoch_0")

    # replace

    MIC_cutoff = args.start_MIC_cutoff
    TOXIN_cutoff = args.start_TOXIN_cutoff

    for feedback_epoch in range(args.feedback_epochs + 13):
        print("feedback_epoch: ", feedback_epoch)

        model.eval()
        lora_df = predict(generate(args, model, ProGen2_tokenizer))
        lora_df.to_csv(args.output_dir + "fb_epoch_" + str(feedback_epoch) + "/result.csv")

        candidates = lora_df[
            (lora_df['MIC_predict'] > MIC_cutoff) 
            & 
            (lora_df['TOXIN_predict'] < TOXIN_cutoff)
        ][['SEQUENCE', 'MIC_predict', 'TOXIN_predict']]
        num_candidates = len(candidates)

        # 逐行处理 real_df，替换符合条件的行
        final_rows = []
        candidate_idx = 0  # 跟踪候选序列的索引

        for _, row in train_dataset.iterrows():
            # 判断是否需要替换当前行
            if ( (row['MIC_predict'] < MIC_cutoff) \
                or \
                (row['TOXIN_predict'] > TOXIN_cutoff) \
                    == 1): # 
                # 需要替换，且候选还有剩余
                if candidate_idx < len(candidates):
                    # 替换为候选序列
                    replaced_row = candidates.iloc[candidate_idx]
                    final_rows.append(replaced_row)
                    candidate_idx += 1
                else:
                    # 候选用完，保留原行（替换数量不足）
                    final_rows.append(row)
            else:
                # 不需要替换，保留原行
                final_rows.append(row)

        train_dataset = pd.DataFrame(final_rows)
        train_dataset.to_csv(args.output_dir + "fb_epoch_" + str(feedback_epoch) + "/train.csv")
        train_dataset = train_dataset.reset_index(drop=True)

        data_loader = _get_train_data_loader(dataset = train_dataset)

        model.train()
        for epoch in range(args.num_train_epochs):
            epoch_loss = 0.0
            step_count = 0
            start_time = time.time()

            optimizer.zero_grad()
            for step, batch in enumerate(data_loader):
                # batch["input_ids"], batch["labels"] 来自 Dataset
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                with autocast(dtype=torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16):
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss / accumulation_steps # HuggingFace 的模型自动计算好了 cross-entropy loss

                epoch_loss += loss.item() * accumulation_steps
                scaler.scale(loss).backward()
                step_count += 1

                if (step + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                # 每 100 步打印一次
                if (step + 1) % 100 == 0:
                    avg_loss = epoch_loss / step_count
                    elapsed = time.time() - start_time
                    print(f"[Epoch {epoch+1}] Step {step+1}/{len(data_loader)} "
                        f" | Avg Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

            avg_epoch_loss = epoch_loss / step_count
            elapsed = time.time() - start_time
            print(f"Feedback_Epoch {epoch+1} finished. Avg Loss = {avg_epoch_loss:.4f} | Time = {elapsed:.1f}s")

        model.save_pretrained("fb_epoch_" + str(feedback_epoch+1))

        MIC_cutoff = MIC_cutoff + (args.target_MIC_cutoff - args.start_MIC_cutoff) / args.feedback_epochs
        TOXIN_cutoff = TOXIN_cutoff + (args.target_TOXIN_cutoff - args.start_TOXIN_cutoff) / args.feedback_epochs if feedback_epoch < args.feedback_epochs else TOXIN_cutoff

    model.eval()
    lora_df = predict(generate(args, model, ProGen2_tokenizer))
    lora_df.to_csv(args.output_dir + "fb_epoch_" + str(args.feedback_epochs) + "/result.csv")


if __name__ == "__main__":
    OUTPUT_PATH = "checkpoints/mul/"

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=OUTPUT_PATH)
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--start_MIC_cutoff", type=float, default=0.70)
    parser.add_argument("--target_MIC_cutoff", type=float, default=0.90)
    parser.add_argument("--start_TOXIN_cutoff", type=float, default=0.55)
    parser.add_argument("--target_TOXIN_cutoff", type=float, default=0.33)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--feedback_epochs", type=int, default=7)
    parser.add_argument("--per_device_train_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--trainable", type=bool, default=True)

    finetuning_and_replace(parser.parse_args())

    print("end")
