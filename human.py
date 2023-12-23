import json
import csv
import numpy as np
import random

def load_human_data():
    with open('json/human.json', 'r') as json_file:
        human_answer = json.load(json_file)
        return human_answer

def human_answer(question, human_data):   
    if question in human_data:
        if human_data[question] == "":
            return "Information unavailable"
        else:
            return f"{human_data[question]['value']}"

def load_human_question():
    with open('json/human_ask.json', 'r') as json_file:
        return json.load(json_file)

def human_question(human_question_data):
    
    question_probabilities = [human_question_data[key]['probability'] for key in human_question_data]
    
    sum_probabilities = sum(question_probabilities)
    normalized_probabilities = [prob / sum_probabilities for prob in question_probabilities]
    
    max_prob = max(normalized_probabilities)
    
    # Count of the highest number
    count_prob = normalized_probabilities.count(max_prob)

    if count_prob > 2:
        question_count = 5
    else:
        question_count = 2
    
    highest_prob_indices = np.argsort(normalized_probabilities)[::-1][:question_count]
    #human_questions = np.random.choice(list(human_question_data.keys()), question_count, p=normalized_probabilities, replace=False)
    human_questions = np.random.choice([list(human_question_data.keys())[i] for i in highest_prob_indices], question_count, replace=False)

    print("=====human_questions=====",human_questions)
    return human_questions
    
def process_app_feature(question):
    prob = random.uniform(0, 1)
    prob_count = ( (prob - 0) / (1 - 0) ) * (5 - 1) + 1
    return round(prob_count)

def save_magic_data(magic_c_data):
    with open('magic-c.json', 'w') as magic_c_json_file:
        json.dump(magic_c_data, magic_c_json_file, indent=4)
