from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
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




if len(sys.argv) != 2:
    print("Usage: python script.py dataset_name")
    print("dataset_name: amazon_beauty, amazon_electronics, amazon_fashion, amazon_garden, amazon_grocery, goodreads, mobilerec, tripadvisor, yelp")
    sys.exit(1)




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

model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path = "models/t5_baseline_prev_interactions/"+sys.argv[1])
model.eval()
model = model.to('cuda')
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", additional_special_tokens=["<|sep|>"])
model.resize_token_embeddings(len(tokenizer))
IGNORE_INDEX = -100



test_raw = pd.read_json(dataset + "/test.jsonl", lines=True)



user_id = []
previous_interactions = []
recommended_item_name = []
turns = []
recommend_indexes = []

for index, row in tqdm(test_raw.iterrows(), total = len(test_raw)):
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
    

df_recommender_test = pd.DataFrame({"user_id": user_id, "previous_interactions":previous_interactions, "recommended_item_name":recommended_item_name, "turns": turns, "recommend_indexes":recommend_indexes})




df_recommender_test = df_recommender_test[(df_recommender_test["recommend_indexes"] != -1) & (df_recommender_test["turns"].apply(lambda x: len(x) > 0))]
df_recommender_test['user_id'] = df_recommender_test['user_id'].str.lower()
df_recommender_test['previous_interactions'] = df_recommender_test['previous_interactions'].str.lower()
df_recommender_test['recommended_item_name'] = df_recommender_test['recommended_item_name'].str.lower()
df_recommender_test['turns'] = df_recommender_test['turns'].apply(lambda x: [s.lower() for s in x])



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






def fix_recommended_items_names(row):
    if row["recommended_item_name"] not in all_items:
        for item in all_items:
            if fuzz.ratio(row["recommended_item_name"], item) > 80:
                return item
        return "uno!™"
    else:
        return row["recommended_item_name"]

df_recommender_test['recommended_item_name'] = df_recommender_test.apply(fix_recommended_items_names, axis=1)






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
recommend_test = []
candidate_items = []
true_candidate_index = []
for _, row in df_recommender_test.iterrows():
    candidates = []
    for index, candidate_item in enumerate(row["candidate_items"]):
        candidates.append(candidate_item)
        if candidate_item == row["recommended_item_name"]:
            true_candidate_index.append(index)

    candidate_items.append(candidates)
    if row["previous_interactions"] is not None:
        prompt = row["previous_interactions"]
    prompt_test.append(prompt)
    recommend_test.append(row["recommended_item_name"])








def chunk(list_of_elements, batch_size): # using this chunk function, we can split our data to multiple batches
  for i in range(0, len(list_of_elements), batch_size):
    yield list_of_elements[i:i+batch_size]

def evaluate_recommender(prompt_test, recommend_test, model, tokenizer, batch_size=8, threshold=70):
  prompt_batches = list(chunk(prompt_test, batch_size))
  generation_batches = list(chunk(recommend_test, batch_size))

  correctly_predicted = []
  for prompt_batch, generation_batch in tqdm(zip(prompt_batches, generation_batches), total = len(generation_batches)):

    inputs = tokenizer(prompt_batch, max_length=1024, truncation=True, padding="max_length", return_tensors="pt") 

    generations_predicted = model.generate(input_ids=inputs["input_ids"].to('cuda'), attention_mask=inputs["attention_mask"].to('cuda'),
                            max_new_tokens=32,
                            num_beams=8,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                            bos_token_id=tokenizer.bos_token_id) # length_penalty=0.8, Set length_penalty to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer sequences.

    decoded_generations = [tokenizer.decode(generation, skip_special_tokens=True, clean_up_tokenization_spaces=True).replace(" app.", "")  for generation in generations_predicted]
    generation_batch = [generation.replace(" app.", "") for generation in generation_batch]
    
    correctly_predicted.extend([1 if fuzz.ratio(predicted, ground_truth) > threshold else 0 for predicted, ground_truth in zip(decoded_generations, generation_batch)])

  return correctly_predicted



correctly_predicted = evaluate_recommender(prompt_test, recommend_test, model, tokenizer, batch_size=4, threshold=95)
success_rate = sum(correctly_predicted) / len(correctly_predicted)
print("success_rate: ", success_rate)




def chunk(list_of_elements, batch_size): # using this chunk function, we can split our data to multiple batches
  for i in range(0, len(list_of_elements), batch_size):
    yield list_of_elements[i:i+batch_size]
    
def convert_to_sublists(numbers, sublist_size):
    return [numbers[i:i+sublist_size] for i in range(0, len(numbers), sublist_size)]

def recommender_rank(prompts, candidate_items, model, tokenizer, batch_size=8):
  model.eval()
  encoder_max_length = 1024
  decoder_max_length = 32
  prompts_tokenized = tokenizer(prompts, max_length=encoder_max_length, truncation=True, padding="max_length", return_tensors="pt")
  
  input_ids_decoder = []
  attention_mask_decoder = []
  input_ids_encoder = []
  attention_mask_encoder  = []
  for index, candidate_item_elements in enumerate(candidate_items):
    candidate_item_elements = [tokenizer.pad_token+element for element in candidate_item_elements] # adding pad token to the beginning of each candidate app
    candidate_items_tokenized = tokenizer(candidate_item_elements, max_length=decoder_max_length, truncation=True, padding="max_length", return_tensors="pt")
    for candidate_item_index in range(len(candidate_item_elements)):
      input_ids_decoder.append(candidate_items_tokenized["input_ids"][candidate_item_index])
      attention_mask_decoder.append(candidate_items_tokenized["attention_mask"][candidate_item_index])
      input_ids_encoder.append(prompts_tokenized["input_ids"][index])
      attention_mask_encoder.append(prompts_tokenized["attention_mask"][index])
  
  input_ids_encoder_batches = list(chunk(input_ids_encoder, batch_size))
  attention_mask_encoder_batches = list(chunk(attention_mask_encoder, batch_size))
  input_ids_decoder_batches = list(chunk(input_ids_decoder, batch_size))
  attention_mask_decoder_batches = list(chunk(attention_mask_decoder, batch_size))
  

  scores = []
  for input_ids_encoder_batch, attention_mask_encoder_batch, input_ids_decoder_batch, attention_mask_decoder_batch in tqdm(zip(input_ids_encoder_batches, attention_mask_encoder_batches, input_ids_decoder_batches, attention_mask_decoder_batches), total = len(input_ids_encoder_batches)):
    decoder_input_ids = torch.stack(input_ids_decoder_batch).to("cuda")
    decoder_attention_mask = torch.stack(attention_mask_decoder_batch).to("cuda")
    input_ids = torch.stack(input_ids_encoder_batch).to("cuda")
    attention_mask = torch.stack(attention_mask_encoder_batch).to("cuda")
    with torch.no_grad():
      model_output = model(decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, 
                           input_ids=input_ids, attention_mask=attention_mask)
    
    logprobs = F.log_softmax(model_output["logits"], dim=-1)[:, :-1, :] # remove the eos token
    output_tokens = decoder_input_ids[:, 1:] # remove the bos token
        
    tokens_logprobs = torch.gather(logprobs, 2, output_tokens[:, :, None]).squeeze(-1).to(torch.float32)
        
    mask = torch.ones(tokens_logprobs.shape, dtype=torch.bool, device="cuda")
    for i, _output in enumerate(output_tokens):
      for j, _token in enumerate(_output):
        if _token == tokenizer.pad_token_id:
          mask[i, j] = False
              
    score = (tokens_logprobs * mask).sum(-1) / mask.sum(-1)
    scores.extend(score.to('cpu').tolist())
    
  # batch_input_representations = torch.cat(batch_input_representations)
  
  scores = convert_to_sublists(scores, len(candidate_items[0]))
  return scores



scores = recommender_rank(prompt_test, candidate_items, model, tokenizer, batch_size=24)


top_k_scores = [top_k_accuracy_score(true_candidate_index, scores, k=k) for k in range(1, 11)]
true_relevance = [[1 if item == index else 0 for item in range(len(candidate_items[0]))] for index in true_candidate_index]
ndgc_scores = [ndcg_score(true_relevance, scores, k=k) for k in range(1, 11)]


print("top-k scores below 1-10")
print(top_k_scores)
print("ndgc scores below 1-10")
print(ndgc_scores)
