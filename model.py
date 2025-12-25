import sys
model_dir = "../progen2-large"
sys.path.append(model_dir)
from peft import LoraConfig, get_peft_model
from peft import PeftModel
from transformers import LogitsProcessor, LogitsProcessorList
from transformers.generation.logits_process import NoRepeatNGramLogitsProcessor
from tokenizers import Tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from configuration_progen import ProGenConfig
from modeling_progen import ProGenForCausalLM

class RepetitionSuppressionProcessor(LogitsProcessor):
    def __init__(self, tokenizer, repeat_threshold=5, target_tokens=('G', 'R'), penalty=5.0):
        """
        repeat_threshold: 连续重复的阈值（超过该数量即惩罚）
        target_tokens: 要惩罚的目标字符
        penalty: logits 惩罚强度（越大抑制越强）
        """
        self.tokenizer = tokenizer
        self.repeat_threshold = repeat_threshold
        self.target_tokens = target_tokens
        self.penalty = penalty
        
        # 获取目标token的ID
        vocab = self.tokenizer.get_vocab()
        self.target_token_ids = [vocab[t] for t in target_tokens if t in vocab]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 取当前批次中最后一个序列
        last_seq = input_ids[0].tolist()
        decoded = self.tokenizer.decode(last_seq, skip_special_tokens=True)

        # 检查末尾是否有连续重复
        if len(decoded) >= self.repeat_threshold:
            for t in self.target_tokens:
                # 统计末尾连续的 t 个数
                count = 0
                for ch in reversed(decoded):
                    if ch == t:
                        count += 1
                    else:
                        break
                if count >= self.repeat_threshold:
                    token_id = self.tokenizer.token_to_id(t)
                    if token_id < scores.size(-1):
                        scores[0, token_id] -= self.penalty
        return scores

ProGen2_tokenizer = Tokenizer.from_file("../progen2-large/tokenizer.json")

processors = LogitsProcessorList([
    RepetitionSuppressionProcessor(ProGen2_tokenizer, repeat_threshold=5, target_tokens=('G', 'R'), penalty=30.0),
    NoRepeatNGramLogitsProcessor(5)
])


class ProGen(nn.Module):

    def __init__(self, args):
        super().__init__()
        config = ProGenConfig.from_pretrained(model_dir)
        self.base_model = ProGenForCausalLM.from_pretrained(model_dir, config=config, torch_dtype="auto")
        self.save_path = args.output_dir
        self.max_length = args.max_length
        if args.trainable == True:
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=["qkv_proj"],
                lora_dropout=args.lora_dropout,
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.base_model, lora_config)
        else:
            self.model = PeftModel.from_pretrained(
                self.base_model, 
                args.output_dir
            )

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()

    def save_pretrained(self, sub_path):
        self.model.save_pretrained(self.save_path + sub_path)

    def generate(self, input_ids, pad_token_id, eos_token_id, bad_words_ids):
        return self.model.generate(
            input_ids=input_ids,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bad_words_ids=bad_words_ids,
            max_length=self.max_length,     # 最长序列长度
            do_sample=True,                 # 是否采样（True 比较多样化，False 就是贪婪解码）
            top_k=15,                       # 只从 top 5 的 token 中采样
            top_p=0.95,                     # nucleus sampling
            temperature=0.95,               # 温度，越大越随机
            logits_processor=processors,    # 对连续长序列的惩罚
        )

    def forward(self, input_ids, labels=None, attention_mask=None):
        """
        input_ids: [batch, seq_len]
        labels: [batch, seq_len] 
        """
        outputs = self.model(
            input_ids=input_ids,
            labels=labels,
        )
        return outputs  # HuggingFace 会返回包含 loss 和 logits 的对象