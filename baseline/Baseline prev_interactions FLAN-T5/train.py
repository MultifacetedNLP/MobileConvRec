from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
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


def get_first_five_words(sentence):
    words = sentence.split()  # Split the sentence into a list of words
    return " ".join(words[:10])  # Join the first 5 words back into a string



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

train_raw = pd.read_json(dataset + "/train.jsonl", lines=True)
valid_raw = pd.read_json(dataset + "/val.jsonl", lines=True)



user_id = []
previous_interactions = []
recommended_item_name = []
turns = []
recommend_indexes = []

for index, row in tqdm(train_raw.iterrows(), total = len(train_raw)):
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
    

df_recommender_train = pd.DataFrame({"user_id": user_id, "previous_interactions":previous_interactions, "recommended_item_name":recommended_item_name, "turns": turns, "recommend_indexes":recommend_indexes})




user_id = []
previous_interactions = []
recommended_item_name = []
turns = []
recommend_indexes = []

for index, row in tqdm(valid_raw.iterrows(), total = len(valid_raw)):
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
    

df_recommender_validation = pd.DataFrame({"user_id": user_id, "previous_interactions":previous_interactions, "recommended_item_name":recommended_item_name, "turns": turns, "recommend_indexes":recommend_indexes})




df_recommender_train = df_recommender_train[(df_recommender_train["recommend_indexes"] != -1) & (df_recommender_train["turns"].apply(lambda x: len(x) > 0))]
df_recommender_train['user_id'] = df_recommender_train['user_id'].str.lower()
df_recommender_train['previous_interactions'] = df_recommender_train['previous_interactions'].str.lower()
df_recommender_train['recommended_item_name'] = df_recommender_train['recommended_item_name'].str.lower()
df_recommender_train['turns'] = df_recommender_train['turns'].apply(lambda x: [s.lower() for s in x])

df_recommender_validation = df_recommender_validation[(df_recommender_validation["recommend_indexes"] != -1) & (df_recommender_validation["turns"].apply(lambda x: len(x) > 0))]
df_recommender_validation['user_id'] = df_recommender_validation['user_id'].str.lower()
df_recommender_validation['previous_interactions'] = df_recommender_validation['previous_interactions'].str.lower()
df_recommender_validation['recommended_item_name'] = df_recommender_validation['recommended_item_name'].str.lower()
df_recommender_validation['turns'] = df_recommender_validation['turns'].apply(lambda x: [s.lower() for s in x])



model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path = "google/flan-t5-base")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", additional_special_tokens=["<|sep|>"])
model.resize_token_embeddings(len(tokenizer))
IGNORE_INDEX = -100



prompt_validation = []
recommend_validation = []
for _, row in df_recommender_validation.iterrows():
    if row["previous_interactions"] is not None:
        prompt = row["previous_interactions"]
    prompt_validation.append(prompt)
    recommend_validation.append(get_first_five_words(row["recommended_item_name"]))





prompt_encodings = tokenizer(prompt_validation, padding='max_length', max_length=1024, truncation=True, return_tensors='pt')
recommend_encodings = tokenizer(recommend_validation, padding='max_length', max_length=32, truncation=True, return_tensors='pt')

labels = recommend_encodings['input_ids']
labels[labels == tokenizer.pad_token_id] = IGNORE_INDEX

dataset = {
    'input_ids': prompt_encodings['input_ids'],
    'attention_mask': prompt_encodings['attention_mask'],
    'labels': labels,
}
dataset_validation = Dataset.from_dict(dataset)



prompt_train = []
recommend_train = []
for _, row in df_recommender_train.iterrows():
    if row["previous_interactions"] is not None:
        prompt = row["previous_interactions"]
    prompt_train.append(prompt)
    recommend_train.append(get_first_five_words(row["recommended_item_name"]))



prompt_encodings = tokenizer(prompt_train, padding='max_length', max_length=1024, truncation=True, return_tensors='pt')
recommend_encodings = tokenizer(recommend_train, padding='max_length', max_length=32, truncation=True, return_tensors='pt')

labels = recommend_encodings['input_ids']
labels[labels == tokenizer.pad_token_id] = IGNORE_INDEX

dataset = {
    'input_ids': prompt_encodings['input_ids'],
    'attention_mask': prompt_encodings['attention_mask'],
    'labels': labels,
}
dataset_train = Dataset.from_dict(dataset)





def data_collator(batch):
    input_ids, attention_mask, labels,  = [], [], []
    for sample in batch:
        input_ids.append(sample['input_ids'])
        attention_mask.append(sample['attention_mask'])
        labels.append(sample['labels'])
    max_encoder_len = max(sum(x) for x in attention_mask)
    max_decoder_len = max(sum([0 if item == IGNORE_INDEX else 1 for item in x]) for x in labels)
    return {
        'input_ids': torch.tensor(input_ids)[:, :max_encoder_len],
        'attention_mask': torch.tensor(attention_mask)[:, :max_encoder_len],
        'labels': torch.tensor(labels)[:, :max_decoder_len]
    }





output_dir = "models/t5_baseline_prev_interactions/" + sys.argv[1]

training_args = TrainingArguments(
    output_dir="models/t5_baseline_prev_interactions/" + sys.argv[1],
    # Set a high max number of epochs (early stopping will terminate training earlier)
    num_train_epochs=10,  # Increased to allow early stopping
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,  # Important for loss (lower is better)
    evaluation_strategy="steps",
    eval_steps=100,  # Evaluate every N steps (use an integer, not a float)
    save_strategy="steps",
    save_steps=100,  # Align save steps with eval steps
    save_total_limit=3,
    gradient_accumulation_steps=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=100,
    weight_decay=0.01,
    disable_tqdm=False,
    push_to_hub=False,
    report_to="none"
)

trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_validation,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )


checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]

if checkpoints:
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()
    
trainer.save_model()
