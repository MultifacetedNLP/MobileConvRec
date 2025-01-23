from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import torch
import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Optional, List
from fuzzywuzzy import fuzz
import numpy as np
import csv

train_raw = pd.read_json("train.jsonl", lines=True)
valid_raw = pd.read_json("val.jsonl", lines=True)

def is_approximate_substring(substring, string, threshold=70):
    for i in range(len(string) - len(substring) + 1):
        window = string[i:i+len(substring)]
        similarity_ratio = fuzz.ratio(substring, window)
        if similarity_ratio >= threshold:
            return True
    return False



user_id = []
previous_interactions = []
recommended_app_name = []
turns = []
recommend_indexes = []

for index, row in train_raw.iterrows():
    user_id.append(row['user_id'])
    prev = row['user_previous_interactions']
    prev_ = [app['app_name'] for app in prev]
    if len(prev_) > 0:
        previous_interactions.append("<|sep|>".join(prev_)+"<|sep|>")
    else:
        previous_interactions.append(None)
    recommended_app_name.append(row['recommended_app']['app_name'])
    dialog_turns = []
    dialog_index = 0
    found_index = False
    for conv in row['turns']:
        if "COMPUTER" in conv:
            turn = 'computer: '+conv['COMPUTER']
            if (row['recommended_app']['app_name'] in turn) and not found_index:
                recommend_indexes.append(dialog_index)
                found_index = True
            dialog_turns.append(turn)
            dialog_index+=1
        if "HUMAN" in conv:
            turn = 'human: '+conv['HUMAN']
            dialog_turns.append(turn)
            dialog_index+=1
    if not found_index: # approximately finding the recommender turn
        for i, dialog_turn in enumerate(dialog_turns):
            if is_approximate_substring(row['recommended_app']['app_name'], dialog_turn):
                recommend_indexes.append(i)
                found_index = True
                break
                    
    if not found_index:
        recommend_indexes.append(-1)
                        
    turns.append(dialog_turns)
    
print(len(user_id))
print(len(previous_interactions))
print(len(recommended_app_name))
print(len(recommend_indexes))
df_recommender_train = pd.DataFrame({"user_id": user_id, "previous_interactions":previous_interactions, "recommended_app_name":recommended_app_name, "turns": turns, "recommend_indexes":recommend_indexes})
print(f"\nnumber of rows: {len(df_recommender_train)}")




user_id = []
previous_interactions = []
recommended_app_name = []
turns = []
recommend_indexes = []

for index, row in valid_raw.iterrows():
    user_id.append(row['user_id'])
    prev = row['user_previous_interactions']
    prev_ = [app['app_name'] for app in prev]
    if len(prev_) > 0:
        previous_interactions.append("<|sep|>".join(prev_)+"<|sep|>")
    else:
        previous_interactions.append(None)
    recommended_app_name.append(row['recommended_app']['app_name'])
    dialog_turns = []
    dialog_index = 0
    found_index = False
    for conv in row['turns']:
        if "COMPUTER" in conv:
            turn = 'computer: '+conv['COMPUTER']
            if (row['recommended_app']['app_name'] in turn) and not found_index:
                recommend_indexes.append(dialog_index)
                found_index = True
            dialog_turns.append(turn)
            dialog_index+=1
        if "HUMAN" in conv:
            turn = 'human: '+conv['HUMAN']
            dialog_turns.append(turn)
            dialog_index+=1
    if not found_index: # approximately finding the recommender turn
        for i, dialog_turn in enumerate(dialog_turns):
            if is_approximate_substring(row['recommended_app']['app_name'], dialog_turn):
                recommend_indexes.append(i)
                found_index = True
                break
                    
    if not found_index:
        recommend_indexes.append(-1)
                        
    turns.append(dialog_turns)
    
print(len(user_id))
print(len(previous_interactions))
print(len(recommended_app_name))
print(len(recommend_indexes))
df_recommender_validation = pd.DataFrame({"user_id": user_id, "previous_interactions":previous_interactions, "recommended_app_name":recommended_app_name, "turns": turns, "recommend_indexes":recommend_indexes})
print(f"\nnumber of rows: {len(df_recommender_validation)}")





df_recommender_train = df_recommender_train[(df_recommender_train["recommend_indexes"] != -1) & (df_recommender_train["turns"].apply(lambda x: len(x) > 0))]
df_recommender_train['user_id'] = df_recommender_train['user_id'].str.lower()
df_recommender_train['previous_interactions'] = df_recommender_train['previous_interactions'].str.lower()
df_recommender_train['recommended_app_name'] = df_recommender_train['recommended_app_name'].str.lower()
df_recommender_train['turns'] = df_recommender_train['turns'].apply(lambda x: [s.lower() for s in x])

df_recommender_validation = df_recommender_validation[(df_recommender_validation["recommend_indexes"] != -1) & (df_recommender_validation["turns"].apply(lambda x: len(x) > 0))]
df_recommender_validation['user_id'] = df_recommender_validation['user_id'].str.lower()
df_recommender_validation['previous_interactions'] = df_recommender_validation['previous_interactions'].str.lower()
df_recommender_validation['recommended_app_name'] = df_recommender_validation['recommended_app_name'].str.lower()
df_recommender_validation['turns'] = df_recommender_validation['turns'].apply(lambda x: [s.lower() for s in x])






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
for _, row in df_recommender_train.iterrows():
    if row["previous_interactions"] is not None:
        prompt = bos + row["previous_interactions"]
    else:
        prompt = bos + "None"
    items_train.append(RecommenderItem(prompt, row["recommended_app_name"] + eos))




items_validation = []
for _, row in df_recommender_validation.iterrows():
    if row["previous_interactions"] is not None:
        prompt = bos + row["previous_interactions"]
    else:
        prompt = bos + "None"
    items_validation.append(RecommenderItem(prompt, row["recommended_app_name"] + eos))


print(items_validation[54])


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
    output_dir="models/baseline_prev_interactions",
    num_train_epochs=5,
    # logging_steps=500,
    # logging_dir=self.cfg.logging_dir,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_strategy="steps",
    evaluation_strategy="steps",
    save_steps=0.3,#self.cfg.save_steps,
    eval_steps=0.3, #self.cfg.eval_steps,
    save_total_limit=3,
    gradient_accumulation_steps=3, #gradient_accumulation_steps,
    per_device_train_batch_size=4, #train_batch_size,
    per_device_eval_batch_size=4, #self.cfg.eval_batch_size,
    warmup_steps=100,
    weight_decay=0.01,
    # dataloader_drop_last=True,
    disable_tqdm=False,
    report_to='none',
    push_to_hub=False
)


trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=recommenderDataset(items_train),
        eval_dataset=recommenderDataset(items_validation), #dm.datasets[DataNames.dev_language_model.value],
        data_collator=training_collator,
    )


trainer.train()
trainer.save_model()
torch.cuda.empty_cache()