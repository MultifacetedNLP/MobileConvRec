import pandas as pd
from tqdm import tqdm
from rapidfuzz import fuzz, process
import csv
import numpy as np
import torch.nn.functional as F
import random
from openai import OpenAI
import re
from sklearn.metrics import top_k_accuracy_score, ndcg_score

random.seed(42)

def openai_chatgpt_models(prompt, model_name):
  client = OpenAI(
      api_key="" 
      )
  try:
        response = client.chat.completions.create(
            model= model_name,  # Corrected model name
            messages=[{"role": "user", "content": prompt}],
            n=1,
            stop=None,
            temperature=0.0,  # Adjust as needed
        )
        return response.choices[0].message.content
  except Exception as e:
        raise Exception(f"Error while calling OpenAI API: {e}")


def open_route_ai_models(prompt, model_name):
  client = OpenAI(
     base_url="https://openrouter.ai/api/v1",
     api_key="",
     )
  
  try:
    response = client.chat.completions.create(
       model= model_name,  # Corrected model name
       messages=[{"role": "user", "content": prompt}],
       stop=None,
       temperature=0.0,  # Adjust as needed
       )
    return response.choices[0].message.content
  except Exception as e:
     raise Exception(f"Error while calling OpenAI API: {e}")
 
def ranks_extractor(response):
    pattern = r'\d+\.\s+(.*?)\s*(?=\d+\.|$)'
    elements = re.findall(pattern, response, re.DOTALL)
    return elements

def get_first_ten_words(sentence):
    words = sentence.split()  # Split the sentence into a list of words
    return " ".join(words[:10])  # Join the first 5 words back into a string

def prepare_shots(path, previous_interactions = False):
    
    input_file = f"{path}/splits/train.jsonl"
    df_recommender_train = pd.read_json(input_file, lines=True)
    for _, row in df_recommender_train.iterrows():
        row["recommended_product"]["product_name"] = row["recommended_product"]["product_name"].lower()
        
    prompts = []
    not_founds = 0
    
    shuffled_train_df = df_recommender_train.sample(frac=1).reset_index(drop=True)

    for _, row in tqdm(shuffled_train_df.iterrows()):
        if previous_interactions:
            if len(row["user_previous_interactions"]) > 0:
                
                sorted_interactions = sorted(
                    row["user_previous_interactions"],
                    key=lambda x: x['date'],
                )
                sorted_interactions = sorted_interactions[min(len(sorted_interactions), 5)-1:]
                
                previous_interactions_items = [get_first_ten_words(previous_interactions["product_name"]) for previous_interactions in sorted_interactions]
                prompt += "Previous Interactions: "
                for previous_interaction in previous_interactions_items:
                    prompt += "'" + previous_interaction + "', "
                prompt += "\n"
            else:
                prompt = "Previous Interactions: No previous interactions" + "\n"
        else:
            prompt = ""
            
        prompt += "Dialog History:\n"
        found = False
        recommended = get_first_ten_words(row["recommended_product"]["product_name"])
        
        for index, turn in enumerate(row["turns"]):
            if "COMPUTER" in turn:
                computer = turn["COMPUTER"]
                
                if fuzz.partial_ratio(recommended, computer.lower()) >= 95:
                    prompt += f"computer: I would recommend the {recommended} product.\n"
                    prompts.append(prompt)
                    found = True
                    break
                else:
                    prompt += "computer: "+ computer + "\n"
            
            if "HUMAN" in turn:
                human = turn["HUMAN"]
                prompt += "human: " + human + "\n"
        
        if not found:
            not_founds += 1
        
        if len(prompts) >= 2:
                break
            
    print(f"Could not find {not_founds}")
    print(f"len(prompt_validation): {len(prompts)}")

    
    prompt = "Dialog Samples:\n"
    prompt += "First Dialog Sample:\n" + prompts[0] + "\n"
    prompt += "Second Dialog Sample:\n" + prompts[1] + "\n"
    
    return prompt
        
    
    


def prepare_dataset(path, dataset_name, previous_interactions = False):
    input_file = f"{path}/splits/test.jsonl"
    df_recommender_test = pd.read_json(input_file, lines=True)
    for _, row in df_recommender_test.iterrows():
        row["recommended_product"]["product_name"] = row["recommended_product"]["product_name"].lower()
        
    apps_training_path = f"{path}/{dataset_name}"

    all_apps = []
    with open(apps_training_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            all_apps.append(get_first_ten_words(row["title"].lower()))
            
    all_apps = list(set(all_apps))
    
    max_existing_length = max(len(item) for item in all_apps)  # Max length in current array
    new_dtype = f'<U{max_existing_length}'

    def candidate_creator(row):
        np.random.seed(row.name)
        selected_values = np.random.choice(np.setdiff1d(all_apps , [get_first_ten_words(row["recommended_product"]["product_name"])]), 24, replace=False).astype(new_dtype) # filter_candidate_apps(row["recommended_product"]["product_name"]) 
        random_position = np.random.randint(0, len(selected_values) + 1)
        
        return np.insert(selected_values, random_position, get_first_ten_words(row["recommended_product"]["product_name"]))

    df_recommender_test['candidate'] = df_recommender_test.apply(lambda row: candidate_creator(row), axis=1)
    
    two_shots = prepare_shots(path, previous_interactions)
    
    prompt_test = []
    recommend_test = []
    candidate_apps = []
    true_candidate_indexes = []
    not_founds = 0

    for _, row in tqdm(df_recommender_test.iterrows(), total=len(df_recommender_test)):
        # creating candidate apps
        candidates = []
        for index, candidate_app in enumerate(row["candidate"].tolist()):
            candidates.append(candidate_app)
            if candidate_app == get_first_ten_words(row["recommended_product"]["product_name"]):
                true_candidate_index = index
                
        prompt = two_shots + "\n"
        found = False
        recommended = get_first_ten_words(row["recommended_product"]["product_name"])
        
        
        # previouse interactions
        if previous_interactions:
            if len(row["user_previous_interactions"]) > 0:
                
                sorted_interactions = sorted(
                    row["user_previous_interactions"],
                    key=lambda x: x['date'],
                )
                sorted_interactions = sorted_interactions[min(len(sorted_interactions), 5)-1:]
                
                previous_interactions_items = [get_first_ten_words(previous_interactions["product_name"]) for previous_interactions in sorted_interactions]
                prompt = "Previous Interactions: "
                for previous_interaction in previous_interactions_items:
                    prompt += "'" + previous_interaction + "', "
                prompt += "\n"
            else:
                prompt += "Previous Interactions: No previous interactions" + "\n"
        
        prompt += "Dialog History:\n"
        
        for index, turn in enumerate(row["turns"]):
            computer = turn["COMPUTER"]
            
            if fuzz.partial_ratio(recommended, computer.lower()) >= 95:
                prompt += "\n"
                prompt += "The products to be ranked are as follows:"
                for app in row["candidate"]:
                    prompt += "'" + app + "', "
                prompt += "\nBased on the dialogue history, please rank the above products in order of relevance to the user's query. Only provide the ranks without additional explanation.\n"
                prompt += "The ranked products: "
                prompt_test.append(prompt)
                recommend_test.append(recommended)
                candidate_apps.append(candidates)
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
    
    return prompt_test, recommend_test, candidate_apps, true_candidate_indexes

path = "/u/spa-d4/grad/mfe261/Projects/MobileConvRec/dataset/amazon_beauty"
dataset_name = "beauty_df.csv"
prompt_test, recommend_test, candidate_apps, true_candidate_indexes = prepare_dataset(path = path, dataset_name = dataset_name, previous_interactions = False)

# truncaate the data for testing
prompt_test = prompt_test[:100]
recommend_test = recommend_test[:100]
candidate_apps = candidate_apps[:100]
true_candidate_indexes = true_candidate_indexes[:100]

scores = []
for prompt, candidates in tqdm(zip(prompt_test, candidate_apps), total=len(prompt_test)):
    candidiate_scores = []
    response = open_route_ai_models(prompt, "openai/gpt-4o-mini")
    # extract the ranks from the response
    ranked_elements = ranks_extractor(response)
    for candidate in candidates:
        match = process.extractOne(candidate, ranked_elements)
        
        if match:
            matched_element, _, matched_index = match
            score = len(ranked_elements) - matched_index
            candidiate_scores.append(score)
        else:
            print("No match found")
            candidiate_scores.append(0)
        
    # normalize the scores
    candidiate_scores = np.array(candidiate_scores)
    if candidiate_scores.sum() == 0:
        candidiate_scores = np.zeros_like(candidiate_scores)
    else:
        candidiate_scores = candidiate_scores / candidiate_scores.sum()
    scores.append(candidiate_scores.tolist())
    
    print(response)

top_k_accuracies = [top_k_accuracy_score(true_candidate_indexes, scores, k=k, labels=list(range(25))).item() for k in range(1, 11)]
print("Top-k Accuracies:", top_k_accuracies)

true_relevance = [[1 if item == index else 0 for item in range(len(candidate_apps[0]))] for index in true_candidate_indexes]
ndcg_scores = [ndcg_score(true_relevance, scores, k=k).item() for k in range(1, 11)]
print("NDCG Scores:", ndcg_scores)