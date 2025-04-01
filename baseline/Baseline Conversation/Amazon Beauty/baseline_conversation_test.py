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

input_file = "/u/spa-d4/grad/mfe261/Projects/MobileConvRec/dataset/amazon_beauty/splits/test.jsonl"
df_recommender_test = pd.read_json(input_file, lines=True)
for _, row in df_recommender_test.iterrows():
    row["recommended_product"]["product_name"] = row["recommended_product"]["product_name"].lower()
    
apps_training_path = "/u/spa-d4/grad/mfe261/Projects/MobileConvRec/dataset/amazon_beauty/beauty_df.csv"

all_apps = []
with open(apps_training_path, 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        all_apps.append(get_first_five_words(row["title"].lower()))
        
all_apps = list(set(all_apps))

def candidate_creator(row):
    np.random.seed(row.name)
    selected_values = np.random.choice(np.setdiff1d(all_apps, [get_first_five_words(row["recommended_product"]["product_name"])]), 24, replace=False) #  
    random_position = np.random.randint(0, len(selected_values) + 1)
    
    return np.insert(selected_values, random_position, get_first_five_words(row["recommended_product"]["product_name"])) 

df_recommender_test['candidate'] = df_recommender_test.apply(lambda row: candidate_creator(row), axis=1)


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

model = GPT2LMHeadModel.from_pretrained("/u/spa-d4/grad/mfe261/Projects/MobileConvRec/models/gpt_models/amazon_beauty").to(device)
model.resize_token_embeddings(len(tokenizer))
model.eval()


prompt_test = []
recommend_test = []
candidate_books = []
true_candidate_indexes = []
not_founds = 0
for _, row in df_recommender_test.iterrows():
    candidates = []
    for index, candidate_book in enumerate(row["candidate"].tolist()):
        candidates.append(candidate_book)
        if candidate_book == get_first_five_words(row["recommended_product"]["product_name"]):
            true_candidate_index = index
    prompt = bos
    found = False
    recommended = get_first_five_words(row["recommended_product"]["product_name"])
    
    for index, turn in enumerate(row["turns"]):
        computer = turn["COMPUTER"]
        
        if fuzz.partial_ratio(recommended, computer.lower()) >= 90:
            prompt += "computer: I would recommend the "
            prompt_test.append(prompt)
            recommend_test.append('<|sep|>' + recommended + eos)
            candidate_books.append(candidates)
            true_candidate_indexes.append(true_candidate_index)
            found = True
            break
        else:
            prompt += "computer: "+ computer + "\n"
        
        if "HUMAN" in turn:
            human = turn["HUMAN"]
            prompt += "human: " + human + "\n"
    
    if not found:
        not_founds += 1

print(f"Could not find {not_founds}")
print(f"Number of prompt: {len(prompt_test)}")
print(f"Number of generations: {len(recommend_test)}")
print(f"Number of candidate apps: {len(candidate_books)}")
print(f"Number of true candidate indexes: {len(true_candidate_indexes)}")


def chunk(list_of_elements, batch_size): # using this chunk function, we can split our data to multiple batches
  for i in range(0, len(list_of_elements), batch_size):
    yield list_of_elements[i:i+batch_size]

def evaluate_recommender(prompts, generations, model, tokenizer, batch_size=8, device=device, threshold=70):
  prompt_batches = list(chunk(prompts, batch_size))
  generation_batches = list(chunk(generations, batch_size))
  max_length=1024
  generation_length = 32
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
    decoded_generations = [tokenizer.decode(generation, skip_special_tokens=True, clean_up_tokenization_spaces=True)  for generation in generations_predicted]
    generation_batch = [generation.replace("<|endoftext|>", "") for generation in generation_batch]
    
    correctly_predicted.extend([1 if fuzz.ratio(predicted, ground_truth) > threshold else 0 for predicted, ground_truth in zip(decoded_generations, generation_batch)])
    
  
  return correctly_predicted

correctly_predicted = evaluate_recommender(prompt_test, recommend_test, model, tokenizer, batch_size=2, device=device, threshold=95)
success_rate = sum(correctly_predicted) / len(correctly_predicted)
print("success_rate: ", success_rate)


def chunk(list_of_elements, batch_size): # using this chunk function, we can split our data to multiple batches
  for i in range(0, len(list_of_elements), batch_size):
    yield list_of_elements[i:i+batch_size]
    
def convert_to_sublists(numbers, sublist_size):
    return [numbers[i:i+sublist_size] for i in range(0, len(numbers), sublist_size)]

def recommender_rank(prompts, candidate_apps, model, tokenizer, batch_size=8, device=device):
  model.eval()
  tokenizer.padding_side='left'
  tokenizer.truncation_side='left'
  max_length = 1024
  prompts_ids = tokenizer(prompts, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
  
  tokenizer.padding_side='right'
  tokenizer.truncation_side='right'
  input_ids = []
  attention_mask = []
  for index, candidate_app_elements in enumerate(candidate_apps):
    candidate_apps_ids = tokenizer(candidate_app_elements, max_length=32, truncation=True, padding="max_length", return_tensors="pt")
    for candidate_app_index in range(len(candidate_app_elements)):
      input_ids.append(torch.cat([prompts_ids["input_ids"][index], candidate_apps_ids["input_ids"][candidate_app_index]]))
      attention_mask.append(torch.cat([prompts_ids["attention_mask"][index], candidate_apps_ids["attention_mask"][candidate_app_index]]))
      
  input_ids_batches = list(chunk(input_ids, batch_size))
  attention_mask_batches = list(chunk(attention_mask, batch_size))

  scores = []
  for input_ids_batch, attention_mask_batch in tqdm(zip(input_ids_batches, attention_mask_batches), total = len(attention_mask_batches)):

    input_ids = torch.stack(input_ids_batch).to(device)
    attention_mask = torch.stack(attention_mask_batch).to(device)
    with torch.no_grad():
      model_output = model(input_ids=input_ids) # attention_mask=attention_mask

    logprobs = F.log_softmax(model_output["logits"], dim=-1)[:, max_length -1:-1, :]
    output_tokens = input_ids[:, max_length:]
    
    tokens_logprobs = torch.gather(logprobs, 2, output_tokens[:, :, None]).squeeze(-1).to(torch.float32)
    
    mask = torch.ones(tokens_logprobs.shape, dtype=torch.bool, device=device)
    for i, _output in enumerate(output_tokens):
      for j, _token in enumerate(_output):
        if _token == tokenizer.pad_token_id:
          mask[i, j] = False
          
    score = (tokens_logprobs * mask).sum(-1) / mask.sum(-1)
    scores.extend(score.to('cpu').tolist())
  
  scores = convert_to_sublists(scores, len(candidate_apps[0]))
  
  return scores

scores = recommender_rank(prompt_test, recommend_test, model, tokenizer, batch_size=4, device=device)

print([top_k_accuracy_score(true_candidate_indexes, scores, k=k) for k in range(1, 11)])


true_relevance = [[1 if item == index else 0 for item in range(len(recommend_test[0]))] for index in true_candidate_indexes]


print([ndcg_score(true_relevance, scores, k=k) for k in range(1, 11)])