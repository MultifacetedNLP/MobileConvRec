import torch
import torch.nn as nn
from transformers import T5PreTrainedModel, AutoModelForSeq2SeqLM, T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutputWithPastAndCrossAttentions
from transformers.models.t5.modeling_t5 import (
    T5LayerSelfAttention,
    T5LayerCrossAttention,
    T5LayerFF,
    T5LayerNorm,
    T5Stack,
)
import copy
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments,EarlyStoppingCallback
import torch
import os
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from fuzzywuzzy import fuzz
import csv
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import top_k_accuracy_score, ndcg_score
import sys
from model import Combined_Model_2a





if len(sys.argv) != 4:
    print("Usage: python script.py dataset_name modelA_path modelB_path")
    print("dataset_name: amazon_beauty, amazon_electronics, amazon_fashion, amazon_garden, amazon_grocery, goodreads, mobilerec, tripadvisor, yelp")
    sys.exit(1)




import os
import random


# Set random seeds for reproducibility
SEED = 112  # You can use any integer value

random.seed(SEED)  # Python random module
np.random.seed(SEED)  # NumPy
torch.manual_seed(SEED)  # PyTorch
torch.cuda.manual_seed_all(SEED)  # For CUDA (if using GPU)

# Ensure deterministic behavior in PyTorch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set environment variable for PyTorch and Hugging Face
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"



modelA_path = sys.argv[2]
modelB_path = sys.argv[3]

modelA = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path = modelA_path)
modelB = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path = modelB_path)
model = Combined_Model_2a(modelA,modelB)
tokenizer1 = AutoTokenizer.from_pretrained("google/flan-t5-base", additional_special_tokens=["computer:", "human:"])
tokenizer2 = AutoTokenizer.from_pretrained("google/flan-t5-base", additional_special_tokens=["<|sep|>"])
IGNORE_INDEX = -100


model.load_state_dict(torch.load("models/combined_2a/" + sys.argv[1]+"/pytorch_model.bin"))
model.to('cuda')

def is_approximate_substring(substring, string, threshold=70):
    for i in range(len(string) - len(substring) + 1):
        window = string[i:i+len(substring)]
        similarity_ratio = fuzz.ratio(substring, window)
        if similarity_ratio >= threshold:
            return True
    return False


dataset = sys.argv[1]

recommended_column_name = ''
name_column_name = ''
if dataset == 'amazon_beauty' or dataset == 'amazon_electronics' or dataset == 'amazon_fashion' or dataset == 'amazon_garden' or dataset == 'amazon_grocery':
    recommended_column_name = 'recommended_product'
    name_column_name = 'product_name'
elif dataset == 'goodreads':
    recommended_column_name = 'recommended_book'
    name_column_name = 'book_name'
elif dataset == 'mobilerec':
    recommended_column_name = 'recommended_app'
    name_column_name = 'app_name'
elif dataset == 'tripadvisor' or dataset == 'yelp':
    recommended_column_name = 'recommended_place'
    name_column_name = 'place_name'
else:
    print("Invalid dataset")
    sys.exit(1)


dataset = 'datasets/' + sys.argv[1]

test_raw = pd.read_json(dataset + "/test.jsonl", lines=True)


user_id = []
previous_interactions = []
recommended_item_name = []
turns = []
recommend_indexes = []

for index, row in test_raw.iterrows():
    user_id.append(row['user_id'])
    prev = row['user_previous_interactions']
    prev_ = [item[name_column_name] for item in prev]
    if len(prev_) > 0:
        previous_interactions.append("<|sep|>".join(prev_)+'<|sep|>')
    else:
        previous_interactions.append(None)
    recommended_item_name.append(row[recommended_column_name][name_column_name])
    dialog_turns = []
    dialog_index = 0
    found_index = False
    for conv in row['turns']:
        if "COMPUTER" in conv:
            turn = 'computer: '+conv['COMPUTER']
            if (row[recommended_column_name][name_column_name] in turn) and not found_index:
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
            if is_approximate_substring(row[recommended_column_name][name_column_name], dialog_turn):
                recommend_indexes.append(i)
                found_index = True
                break
                    
    if not found_index:
        recommend_indexes.append(-1)
                        
    turns.append(dialog_turns)
    
print(len(user_id))
print(len(previous_interactions))
print(len(recommended_item_name))
print(len(recommend_indexes))
df_recommender_test = pd.DataFrame({"user_id": user_id, "previous_interactions":previous_interactions, "recommended_item_name":recommended_item_name, "turns": turns, "recommend_indexes":recommend_indexes})
print(f"\nnumber of rows: {len(df_recommender_test)}")


df_recommender_test = df_recommender_test[(df_recommender_test["recommend_indexes"] != -1) & (df_recommender_test["turns"].apply(lambda x: len(x) > 0))]
df_recommender_test['user_id'] = df_recommender_test['user_id'].str.lower()
df_recommender_test['previous_interactions'] = df_recommender_test['previous_interactions'].str.lower()
df_recommender_test['recommended_item_name'] = df_recommender_test['recommended_item_name'].str.lower()
df_recommender_test['turns'] = df_recommender_test['turns'].apply(lambda x: [s.lower() for s in x])

def get_first_five_words(sentence):
    words = sentence.split()  # Split the sentence into a list of words
    return " ".join(words[:10])  # Join the first 5 words back into a string


dataset = sys.argv[1]
name_col = ""
if dataset == 'amazon_beauty' or dataset == 'amazon_electronics' or dataset == 'amazon_fashion' or dataset == 'amazon_garden' or dataset == 'amazon_grocery':
    name_col = 'title'
elif dataset == 'goodreads':
    name_col = 'title'
elif dataset == 'mobilerec':
    name_col = 'app_name'
elif dataset == 'tripadvisor' or dataset == 'yelp':
    name_col = 'name'
else:
    print("Invalid dataset")
    sys.exit(1)


dataset = 'datasets/' + sys.argv[1]
items_path = dataset + "/item_info.csv"
print(items_path)

all_items = []
with open(items_path, 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        all_items.append(row[name_col].lower())



# Ensure that you've already defined 'apps_training_path' to point to your data file
cols = [name_col]
df_item = pd.read_csv(items_path, usecols=cols,encoding="utf-8")
df_items = df_item.applymap(lambda x: x.lower() if isinstance(x, str) else x)



def filter_candidate_apps(rec_item_name):
    df_filtered = df_items.drop_duplicates(subset=[name_col])
    df_filtered = df_filtered[df_filtered[name_col] != rec_item_name]
    candidate_apps = set(df_filtered.sample(n=25, random_state=42)[name_col])
    

    return list(candidate_apps)  # Converting back to list if needed for downstream processes






def candidate_creator(row):
    np.random.seed(row.name)
    
    # Get candidate items
    candidates = list(np.setdiff1d(filter_candidate_apps(row["recommended_item_name"]), [row["recommended_item_name"]]))

    # Select 24 items
    selected_values = np.random.choice(candidates, 24, replace=False).tolist()  # Convert NumPy array to list

    # Randomly insert the recommended item (using list insert instead of np.insert)
    random_position = np.random.randint(0, len(selected_values) + 1)
    selected_values.insert(random_position, row["recommended_item_name"])  # Python list insert
    
    return selected_values  # Return as a list instead of a NumPy array

df_recommender_test['candidate_items'] = df_recommender_test.apply(lambda row: candidate_creator(row), axis=1)




prompt_test = []
interactions_test = []
recommend_test = []
candidate_books = []
true_candidate_indexes = []
not_founds = 0
for _, row in df_recommender_test.iterrows():
    candidates = []
    for index, candidate_book in enumerate(row["candidate_items"]):
        candidates.append(candidate_book)
        if candidate_book == get_first_five_words(row["recommended_item_name"]):
            true_candidate_index = index
    prompt = ""
    
    found = False
    recommended = get_first_five_words(row["recommended_item_name"])
    if row["previous_interactions"] is not None:
        interactions = row["previous_interactions"]
    
    for index, turn in enumerate(row["turns"]):
        computer = turn
        
        if fuzz.partial_ratio(recommended, computer.lower()) >= 95:
            prompt += "computer: I would recommend the "
            prompt_test.append(prompt)
            recommend_test.append(recommended)
            interactions_test.append(interactions)
            candidate_books.append(candidates)
            true_candidate_indexes.append(true_candidate_index)
            found = True
            break
        else:
            prompt += computer + "\n"
        
        if "HUMAN" in turn:
            human = turn
            prompt += human + "\n"
    
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

def evaluate_recommender(prompt_test,interactions_test, recommend_test, model, tokenizer1,tokenizer2, batch_size=8, threshold=70):
  prompt_batches = list(chunk(prompt_test, batch_size))
  interactions_batches = list(chunk(interactions_test, batch_size))
  generation_batches = list(chunk(recommend_test, batch_size))

  correctly_predicted = []
  for prompt_batch, generation_batch,interactions_batch in tqdm(zip(prompt_batches, generation_batches,interactions_batches), total = len(generation_batches)):

    inputs1 = tokenizer1(prompt_batch, max_length=1024, truncation=True, padding="max_length", return_tensors="pt") 
    inputs2 = tokenizer2(interactions_batch, max_length=1024, truncation=True, padding="max_length", return_tensors="pt") 

    generations_predicted = model.generate(input_idsA=inputs1["input_ids"].to('cuda'), attention_maskA=inputs1["attention_mask"].to('cuda'),input_idsB=inputs2["input_ids"].to('cuda'), attention_maskB=inputs2["attention_mask"].to('cuda'),
                            max_length=32,
                            num_beams=1) # length_penalty=0.8, Set length_penalty to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer sequences.

    decoded_generations = [tokenizer1.decode(generation, skip_special_tokens=True, clean_up_tokenization_spaces=True).replace(" app.", "")  for generation in generations_predicted]
    generation_batch = [generation.replace(" app.", "") for generation in generation_batch]
    
    correctly_predicted.extend([1 if fuzz.ratio(predicted, ground_truth) > threshold else 0 for predicted, ground_truth in zip(decoded_generations, generation_batch)])

  return correctly_predicted


correctly_predicted = evaluate_recommender(prompt_test,interactions_test, recommend_test, model, tokenizer1,tokenizer2, batch_size=1, threshold=70)
success_rate = sum(correctly_predicted) / len(correctly_predicted)
print("success_rate: ", success_rate)


def recommender_rank(promptsA, promptsB, candidate_apps, model, tokenizer1, tokenizer2, batch_size=8):
    model.eval()
    encoder_max_length = 1024
    decoder_max_length = 32

    # Tokenize both encoder inputs with appropriate tokenizers
    encoderA_inputs = tokenizer1(promptsA, max_length=encoder_max_length, 
                                truncation=True, padding="max_length", return_tensors="pt")
    encoderB_inputs = tokenizer2(promptsB, max_length=encoder_max_length,
                                truncation=True, padding="max_length", return_tensors="pt")

    # Prepare all inputs for batch processing
    all_inputs = {
        'input_idsA': [],
        'attention_maskA': [],
        'input_idsB': [],
        'attention_maskB': [],
        'labels': []
    }

    # Process each prompt pair and its candidates
    for idx, candidates in enumerate(candidate_apps):
        # Tokenize candidates as labels using decoder tokenizer
        candidate_inputs = tokenizer1(candidates, max_length=decoder_max_length,
                                     truncation=True, padding="max_length", return_tensors="pt")
        
        # For each candidate, add corresponding encoder inputs
        for candidate_idx in range(len(candidates)):
            all_inputs['input_idsA'].append(encoderA_inputs["input_ids"][idx])
            all_inputs['attention_maskA'].append(encoderA_inputs["attention_mask"][idx])
            all_inputs['input_idsB'].append(encoderB_inputs["input_ids"][idx])
            all_inputs['attention_maskB'].append(encoderB_inputs["attention_mask"][idx])
            all_inputs['labels'].append(candidate_inputs["input_ids"][candidate_idx])

    # Batch all inputs (FIX APPLIED HERE)
    batched_inputs = {
        k: list(chunk(values, batch_size))  # Convert generators to lists
        for k, values in all_inputs.items()
    }

    scores = []
    for batch_idx in tqdm(range(len(batched_inputs['input_idsA']))):
        # Prepare batch (now working with lists of tensors)
        batch = {
            'input_idsA': torch.stack(batched_inputs['input_idsA'][batch_idx]).to("cuda"),
            'attention_maskA': torch.stack(batched_inputs['attention_maskA'][batch_idx]).to("cuda"),
            'input_idsB': torch.stack(batched_inputs['input_idsB'][batch_idx]).to("cuda"),
            'attention_maskB': torch.stack(batched_inputs['attention_maskB'][batch_idx]).to("cuda"),
            'labels': torch.stack(batched_inputs['labels'][batch_idx]).to("cuda")
        }

        with torch.no_grad():
            outputs = model(**batch)

        # Calculate scores same as before
        logits = outputs.logits
        logprobs = F.log_softmax(logits, dim=-1)[:, :-1, :]
        output_tokens = batch['labels'][:, 1:]  # Skip decoder_start_token_id
        
        # Mask padding tokens
        mask = (output_tokens != tokenizer1.pad_token_id).float()
        tokens_logprobs = torch.gather(logprobs, 2, output_tokens.unsqueeze(-1)).squeeze(-1)
        sequence_scores = (tokens_logprobs * mask).sum(dim=-1) / mask.sum(dim=-1)
        
        scores.extend(sequence_scores.cpu().tolist())

    # Group scores by original prompts
    return [scores[i:i+len(candidate_apps[0])] for i in range(0, len(scores), len(candidate_apps[0]))]



scores = recommender_rank(prompt_test,interactions_test, candidate_books, model, tokenizer1,tokenizer2, batch_size=8)
top_k = [top_k_accuracy_score(true_candidate_indexes, scores, k=k) for k in range(1, 11)]
print("Top-k accuracy:")
print(top_k)
true_relevance = [[1 if item == index else 0 for item in range(len(candidate_books[0]))] for index in true_candidate_indexes]
ndgc = [ndcg_score(true_relevance, scores, k=k) for k in range(1, 11)]
print("NDCG:")
print(ndgc)