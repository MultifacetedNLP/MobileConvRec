import os

from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
import torch
import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Optional, List
from fuzzywuzzy import fuzz
import evaluate
import csv
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import top_k_accuracy_score, ndcg_score

def get_first_five_words(sentence):
    words = sentence.split()  # Split the sentence into a list of words
    return " ".join(words[:10])  # Join the first 5 words back into a string

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


input_file = "/u/spa-d4/grad/mfe261/Projects/MobileConvRec/dataset/amazon_beauty/splits/train.jsonl"
df_recommender_train = pd.read_json(input_file, lines=True)
for _, row in df_recommender_train.iterrows():
    row["recommended_product"]["product_name"] = row["recommended_product"]["product_name"].lower()


input_file = "/u/spa-d4/grad/mfe261/Projects/MobileConvRec/dataset/amazon_beauty/splits/val.jsonl"
df_recommender_validation = pd.read_json(input_file, lines=True)
for _, row in df_recommender_validation.iterrows():
    row["recommended_product"]["product_name"] = row["recommended_product"]["product_name"].lower()

model_checkpoint = "gpt2"
bos = '<|startoftext|>'
eos = '<|endoftext|>'
pad = '<|pad|>'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint, bos_token=bos, eos_token=eos, pad_token=pad, additional_special_tokens=["<|sep|>","computer:", "human:"])

model = GPT2LMHeadModel.from_pretrained(model_checkpoint).to(device)
model.resize_token_embeddings(len(tokenizer))
model_max_length=1024


@dataclass
class RecommenderItem:
    prompt: str
    generation: Optional[str] = None
    
class recommenderDataset(Dataset):
    def __init__(self, data: List[RecommenderItem]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> RecommenderItem:
        return self.data[idx]


items_train = []
not_founds = 0

for _, row in tqdm(df_recommender_train.iterrows(), total=len(df_recommender_train)):
    prompt = bos
    found = False
    recommended = get_first_five_words(row["recommended_product"]["product_name"])
    
    for index, turn in enumerate(row["turns"]):
        if "COMPUTER" in turn:
            computer = turn["COMPUTER"]

            if fuzz.partial_ratio(recommended, computer.lower()) >= 90:
                prompt += "computer: I would recommend the "
                items_train.append(RecommenderItem(prompt, recommended + eos))
                found = True
                break
            else:
                prompt += "computer: " + computer + "\n"
        
        if "HUMAN" in turn:
            human = turn["HUMAN"]
            prompt += "human: " + human + "\n"
    
    if not found:
        not_founds += 1

print(f"Could not find {not_founds}")
print(f" number of items: {len(items_train)}")


items_validation = []
not_founds = 0

for _, row in tqdm(df_recommender_validation.iterrows(), total=len(df_recommender_validation)):
    prompt = bos
    found = False
    recommended = get_first_five_words(row["recommended_product"]["product_name"])
    
    for index, turn in enumerate(row["turns"]):
        computer = turn["COMPUTER"]
        
        if fuzz.partial_ratio(recommended, computer.lower()) >= 90:
            prompt += "computer: I would recommend the "
            items_validation.append(RecommenderItem(prompt, recommended + eos))
            found = True
            break
        else:
            prompt += "computer: " + computer + "\n"
        
        if "HUMAN" in turn:
            human = turn["HUMAN"]
            prompt += "human: " + human + "\n"
    
    if not found:
        not_founds += 1
        
print(f"Could not find {not_founds}")
print(f" number of items: {len(items_validation)}")


def training_collator(batch: list[recommenderDataset]): # for training a language model
    input_ids = []
    attention_masks = []
    labels = []
    for item in batch:
        prompt_tokens = tokenizer.encode(item.prompt, return_tensors="pt")[0] 
        generation_tokens = tokenizer.encode(item.generation, return_tensors="pt")[0]
        prompt_len = len(prompt_tokens)
        generation_len = len(generation_tokens)
        unused_len = model_max_length - prompt_len - generation_len
        # handling case when input is greater than tokenizer length.
        if unused_len < 0:
            prompt_start_tokens = prompt_tokens[:1]
            trimmed_prompt = prompt_tokens[unused_len * -1 + 1 :] # TODO: you could delete the prompt to reach the first |beginuser| token
            prompt_tokens = torch.cat(
                [prompt_start_tokens, trimmed_prompt], axis=0
            )
            prompt_len = len(prompt_tokens)
            unused_len = 0
        pad = torch.full([unused_len], tokenizer.pad_token_id)
        input_tokens = torch.cat(
            [prompt_tokens, generation_tokens, pad]
        )
        label = torch.cat(
            [
                torch.full(
                    [prompt_len],
                    -100,
                ),
                generation_tokens,
                torch.full([unused_len], -100),
            ]
        )
        attention_mask = torch.cat(
            [
                torch.full([prompt_len + generation_len], 1),
                torch.full([unused_len], 0),
            ]
        )
        input_ids.append(input_tokens)
        attention_masks.append(attention_mask)
        labels.append(label)

    out = {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(labels),
    }

    return out


training_args = TrainingArguments(
    output_dir="/u/spa-d4/grad/mfe261/Projects/MobileConvRec/models/gpt_models/amazon_beauty",
    num_train_epochs=10,
    # logging_steps=500,
    # logging_dir=self.cfg.logging_dir,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,  # Important for loss (lower is better)
    save_strategy="steps",
    evaluation_strategy="steps",
    save_steps=100,#self.cfg.save_steps,
    eval_steps=100,  #self.cfg.eval_steps,
    save_total_limit=3,
    gradient_accumulation_steps=3, #gradient_accumulation_steps,
    per_device_train_batch_size=4, #train_batch_size,
    per_device_eval_batch_size=4, #self.cfg.eval_batch_size,
    warmup_steps=100,
    weight_decay=0.01,
    # dataloader_drop_last=True,
    disable_tqdm=False,
    push_to_hub=False,
    report_to="none"
)


trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=recommenderDataset(items_train),
        eval_dataset=recommenderDataset(items_validation), #dm.datasets[DataNames.dev_language_model.value],
        data_collator=training_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

trainer.train()
trainer.save_model()