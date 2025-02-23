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



test_raw = pd.read_json("test.jsonl", lines=True)


def is_approximate_substring(substring, string, threshold=70):
    for i in range(len(string) - len(substring) + 1):
        window = string[i:i+len(substring)]
        similarity_ratio = fuzz.ratio(substring, window)
        if similarity_ratio >= threshold:
            return True
    return False


user_id = []
previous_interactions = []
recommended_place_name = []
turns = []
recommend_indexes = []

for index, row in test_raw.iterrows():
    user_id.append(row['user_id'])
    prev = row['user_previous_interactions']
    prev_ = [place['place_name'] for place in prev]
    if len(prev_) > 0:
        previous_interactions.append("<|sep|>".join(prev_)+"<|sep|>")
    else:
        previous_interactions.append(None)
    recommended_place_name.append(row['recommended_place']['place_name'])
    dialog_turns = []
    dialog_index = 0
    found_index = False
    for conv in row['turns']:
        if "COMPUTER" in conv:
            turn = 'computer: '+conv['COMPUTER']
            if (row['recommended_place']['place_name'] in turn) and not found_index:
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
            if is_approximate_substring(row['recommended_place']['place_name'], dialog_turn):
                recommend_indexes.append(i)
                found_index = True
                break
                    
    if not found_index:
        recommend_indexes.append(-1)
                        
    turns.append(dialog_turns)
    
print(len(user_id))
print(len(previous_interactions))
print(len(recommended_place_name))
print(len(recommend_indexes))
df_recommender_test = pd.DataFrame({"user_id": user_id, "previous_interactions":previous_interactions, "recommended_place_name":recommended_place_name, "turns": turns, "recommend_indexes":recommend_indexes})
print(f"\nnumber of rows: {len(df_recommender_test)}")


df_recommender_test = df_recommender_test[(df_recommender_test["recommend_indexes"] != -1) & (df_recommender_test["turns"].apply(lambda x: len(x) > 0))]
df_recommender_test['user_id'] = df_recommender_test['user_id'].str.lower()
df_recommender_test['previous_interactions'] = df_recommender_test['previous_interactions'].str.lower()
df_recommender_test['recommended_place_name'] = df_recommender_test['recommended_place_name'].str.lower()
df_recommender_test['turns'] = df_recommender_test['turns'].apply(lambda x: [s.lower() for s in x])


model_checkpoint = "models/baseline_prev_interactions"
bos = '<|startoftext|>'
eos = '<|endoftext|>'
pad = '<|pad|>'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token=bos, eos_token=eos, pad_token=pad, additional_special_tokens=["<|sep|>","computer:", "human:"],padding_side='left')
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
    



items_test = []
for _, row in df_recommender_test.iterrows():
    if row["previous_interactions"] is not None:
        prompt = bos + row["previous_interactions"]
        items_test.append(RecommenderItem(prompt, row["recommended_place_name"] + eos))
    





def chunk(list_of_elements, batch_size): # using this chunk function, we can split our data to multiple batches
  for i in range(0, len(list_of_elements), batch_size):
    yield list_of_elements[i:i+batch_size]

def evaluate_recommender(dataset, model, tokenizer, batch_size=8, device=device, threshold=70):
  prompt_arr = [data.prompt for data in dataset]
  generation_arr = [data.generation for data in dataset]
  prompt_batches = list(chunk(prompt_arr, batch_size))
  generation_batches = list(chunk(generation_arr, batch_size))
  max_length=992
  generation_length = 32
  print(len(dataset))
  correctly_predicted = []

  for prompt_batch, generation_batch in tqdm(zip(prompt_batches, generation_batches), total = len(generation_batches)):

    inputs = tokenizer(prompt_batch, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt") 

    generations_predicted = model.generate(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device),
                            max_new_tokens=generation_length,
                            num_beams=8,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                            bos_token_id=tokenizer.bos_token_id) # length_penalty=0.8, Set length_penalty to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer sequences.

    generations_predicted = generations_predicted[:, max_length:] # we only need the generation part, not the prompt part.
    decoded_generations = [tokenizer.decode(generation, skip_special_tokens=True, clean_up_tokenization_spaces=True) for generation in generations_predicted]
    print(decoded_generations)
    generation_batch = generation_batch
    
    correctly_predicted.extend([1 if fuzz.ratio(predicted, ground_truth) > threshold else 0 for predicted, ground_truth in zip(decoded_generations, generation_batch)])


  return correctly_predicted



correctly_predicted = evaluate_recommender(recommenderDataset(items_test), model, tokenizer, batch_size=4, device=device,threshold=50)
success_rate = sum(correctly_predicted) / len(correctly_predicted)
print("success_rate: ", success_rate)




