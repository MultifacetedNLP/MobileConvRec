from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments,EarlyStoppingCallback
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
import torch.nn.functional as F
from sklearn.metrics import top_k_accuracy_score, ndcg_score
import evaluate


train_raw = pd.read_json("datasets/mobile rec/train.jsonl", lines=True)
valid_raw = pd.read_json("datasets/mobile rec/val.jsonl", lines=True)


def is_approximate_substring(substring, string, threshold=70):
    for i in range(len(string) - len(substring) + 1):
        window = string[i:i+len(substring)]
        similarity_ratio = fuzz.ratio(substring, window)
        if similarity_ratio >= threshold:
            return True
    return False



user_id = []
previous_interactions = []
recommended_product_name = []
turns = []
recommend_indexes = []

for index, row in train_raw.iterrows():
    user_id.append(row['user_id'])
    prev = row['user_previous_interactions']
    prev_ = [(product['app_name']) for product in prev]
    if len(prev_) > 0:
        previous_interactions.append("<|sep|>".join(prev_)+"<|sep|>")
    else:
        previous_interactions.append(None)
    recommended_product_name.append(row['recommended_app']['app_name'])
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
print(len(recommended_product_name))
print(len(recommend_indexes))
df_recommender_train = pd.DataFrame({"user_id": user_id, "previous_interactions":previous_interactions, "recommended_product_name":recommended_product_name, "turns": turns, "recommend_indexes":recommend_indexes})
print(f"\nnumber of rows: {len(df_recommender_train)}")



user_id = []
previous_interactions = []
recommended_product_name = []
turns = []
recommend_indexes = []

for index, row in valid_raw.iterrows():
    user_id.append(row['user_id'])
    prev = row['user_previous_interactions']
    prev_ = [(product['app_name']) for product in prev]
    if len(prev_) > 0:
        previous_interactions.append("<|sep|>".join(prev_)+"<|sep|>")
    else:
        previous_interactions.append(None)
    recommended_product_name.append(row['recommended_app']['app_name'])
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
print(len(recommended_product_name))
print(len(recommend_indexes))
df_recommender_validation = pd.DataFrame({"user_id": user_id, "previous_interactions":previous_interactions, "recommended_product_name":recommended_product_name, "turns": turns, "recommend_indexes":recommend_indexes})
print(f"\nnumber of rows: {len(df_recommender_validation)}")



df_recommender_train = df_recommender_train[(df_recommender_train["recommend_indexes"] != -1) & (df_recommender_train["turns"].apply(lambda x: len(x) > 0))]
df_recommender_train['user_id'] = df_recommender_train['user_id'].str.lower()
df_recommender_train['previous_interactions'] = df_recommender_train['previous_interactions'].str.lower()
df_recommender_train['recommended_product_name'] = df_recommender_train['recommended_product_name'].str.lower()
df_recommender_train['turns'] = df_recommender_train['turns'].apply(lambda x: [s.lower() for s in x])

df_recommender_validation = df_recommender_validation[(df_recommender_validation["recommend_indexes"] != -1) & (df_recommender_validation["turns"].apply(lambda x: len(x) > 0))]
df_recommender_validation['user_id'] = df_recommender_validation['user_id'].str.lower()
df_recommender_validation['previous_interactions'] = df_recommender_validation['previous_interactions'].str.lower()
df_recommender_validation['recommended_product_name'] = df_recommender_validation['recommended_product_name'].str.lower()
df_recommender_validation['turns'] = df_recommender_validation['turns'].apply(lambda x: [s.lower() for s in x])


model_checkpoint = "gpt2"
bos = '<|startoftext|>'
eos = '<|endoftext|>'
pad = '<|pad|>'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint, bos_token=bos, eos_token=eos, pad_token=pad, additional_special_tokens=["<|sep|>","computer:", "human:"])
model = GPT2LMHeadModel.from_pretrained(model_checkpoint).to(device)
model.resize_token_embeddings(len(tokenizer))
model_max_length=512


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
    prompt = bos
    for index, turn in enumerate(row["turns"]):
        prompt += turn + "\n"
        if (index + 1) < len(row["turns"]) and index % 2 == 1: # only add the training sample where the computer is speaking
            items_train.append(RecommenderItem(prompt, row["turns"][index + 1] + eos))



items_validation = []
for _, row in df_recommender_validation.iterrows():
    prompt = bos
    for index, turn in enumerate(row["turns"]):
        prompt += turn + "\n"
        if (index + 1) < len(row["turns"]) and index % 2 == 1: # only add the training sample where the computer is speaking
            items_validation.append(RecommenderItem(prompt, row["turns"][index + 1] + eos))




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
    output_dir="models/conv_gpt2_responder",
    num_train_epochs=10,
    logging_steps=100,
    # logging_dir=self.cfg.logging_dir,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_strategy="steps",
    evaluation_strategy="steps",
    save_steps=100,#self.cfg.save_steps,
    eval_steps=100, #self.cfg.eval_steps,
    save_total_limit=3,
    gradient_accumulation_steps=2, #gradient_accumulation_steps,
    per_device_train_batch_size=6, #train_batch_size,
    per_device_eval_batch_size=6, #self.cfg.eval_batch_size,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )


trainer.train()
trainer.save_model()
torch.cuda.empty_cache()











test_raw = pd.read_json("datasets/mobile rec/test.jsonl", lines=True)


def is_approximate_substring(substring, string, threshold=70):
    for i in range(len(string) - len(substring) + 1):
        window = string[i:i+len(substring)]
        similarity_ratio = fuzz.ratio(substring, window)
        if similarity_ratio >= threshold:
            return True
    return False


user_id = []
previous_interactions = []
recommended_product_name = []
turns = []
recommend_indexes = []

for index, row in test_raw.iterrows():
    user_id.append(row['user_id'])
    prev = row['user_previous_interactions']
    prev_ = [(product['app_name']) for product in prev]
    if len(prev_) > 0:
        previous_interactions.append("<|sep|>".join(prev_)+"<|sep|>")
    else:
        previous_interactions.append(None)
    recommended_product_name.append(row['recommended_app']['app_name'])
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
print(len(recommended_product_name))
print(len(recommend_indexes))
df_recommender_test = pd.DataFrame({"user_id": user_id, "previous_interactions":previous_interactions, "recommended_product_name":recommended_product_name, "turns": turns, "recommend_indexes":recommend_indexes})
print(f"\nnumber of rows: {len(df_recommender_test)}")



df_recommender_test = df_recommender_test[(df_recommender_test["recommend_indexes"] != -1) & (df_recommender_test["turns"].apply(lambda x: len(x) > 0))]
df_recommender_test['user_id'] = df_recommender_test['user_id'].str.lower()
df_recommender_test['previous_interactions'] = df_recommender_test['previous_interactions'].str.lower()
df_recommender_test['recommended_product_name'] = df_recommender_test['recommended_product_name'].str.lower()
df_recommender_test['turns'] = df_recommender_test['turns'].apply(lambda x: [s.lower() for s in x])




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
    


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bos = '<|startoftext|>'
eos = '<|endoftext|>'
pad = '<|pad|>'

tokenizer = GPT2Tokenizer.from_pretrained("gpt2", bos_token=bos, eos_token=eos, pad_token=pad, additional_special_tokens=["<|sep|>", "computer:", "human:"], padding_side='left')
model.eval()



prompt_test = []
recommend_test = []


for _, row in df_recommender_test.iterrows():
    prompt = bos
    for index, turn in enumerate(row["turns"]):
        prompt += turn + "\n"
        if (index + 1) < len(row["turns"]) and index % 2 == 1:
            prompt_test.append(prompt)
            recommend_test.append(row["turns"][index + 1]+eos)
    
    
    
print(f"Number of prompt: {len(prompt_test)}")
print(f"Number of generations: {len(recommend_test)}")



bleu = evaluate.load("bleu")


def chunk(list_of_elements, batch_size): # using this chunk function, we can split our data to multiple batches
  for i in range(0, len(list_of_elements), batch_size):
    yield list_of_elements[i:i+batch_size]

def evaluate_responder(prompt,generation, model, tokenizer, batch_size=8, device=device, bleu=bleu):
  prompt_arr = prompt
  generation_arr = generation
  prompt_batches = list(chunk(prompt_arr, batch_size))
  generation_batches = list(chunk(generation_arr, batch_size))
  prompt_size = 384
  generation_size = 128
  predictions = []
  references = []
  for prompt_batch, generation_batch in tqdm(zip(prompt_batches, generation_batches), total = len(generation_batches)):

    inputs = tokenizer(prompt_batch, max_length=prompt_size, truncation=True, padding="max_length", return_tensors="pt") 

    generations_predicted = model.generate(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device),
                            max_new_tokens=generation_size,
                            num_beams=8,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                            bos_token_id=tokenizer.bos_token_id) # length_penalty=0.8, Set length_penalty to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer sequences.

    generations_predicted = generations_predicted[:, prompt_size:] # we only need the generation part, not the prompt part.
    decoded_generations = [tokenizer.decode(generation, clean_up_tokenization_spaces=True).replace(tokenizer.pad_token, "").replace(tokenizer.eos_token, "")   for generation in generations_predicted]
    generation_batch = [generation.replace(tokenizer.eos_token, "") for generation in generation_batch]
    
    predictions.extend(decoded_generations)
    references.extend([[gen] for gen in generation_batch])

  results = bleu.compute(predictions=predictions, references=references)
  
  return results


results = evaluate_responder(prompt_test,recommend_test, model, tokenizer, batch_size=4, device=device, bleu=bleu)
print("results: ", results)