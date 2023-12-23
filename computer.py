import numpy as np
import json
import pandas as pd
import random

def load_computer_data():
    with open('json/computer.json', 'r') as json_file:
        return json.load(json_file)


def compute_question_count(reviewId):
    df = pd.read_csv('data/final_review_data_updated.csv')
    row = df['reviewId'] == reviewId
    app_aspect_count = df.loc[row, 'aspect_count'].iloc[0]
    max_aspect_count = df["aspect_count"].max()
    random_number = random.uniform(0.5, 1)   
    aspect_probabilty = random_number*(app_aspect_count/max_aspect_count)
    aspect_prob_count = ( (aspect_probabilty - 0) / (1 - 0) ) * (10 - 5) + 5
    return aspect_prob_count

def sample_questions(computer_data, reviewId):

    question_count = compute_question_count(reviewId)
    question_probabilities = [computer_data[key]['probability'] for key in computer_data]
    sum_probabilities = sum(question_probabilities)
    normalized_probabilities = [prob / sum_probabilities for prob in question_probabilities]
    sampled_questions = np.random.choice(list(computer_data.keys()), round(question_count), p=normalized_probabilities, replace=False)
    return sampled_questions

def load_computer_answer_data():
    with open('json/computer_response.json', 'r') as json_file:
        return json.load(json_file)

def computer_answer(question, computer_ans):
    if question in computer_ans:       
        if computer_ans[question] == "":
            return "Information unavailable"
        else:
            return f"{computer_ans[question]['value']}"

    elif (question == "data_collected_detail" or question == "data_shared_detail" or question == "permission_detail"):
        question = question.replace("_detail", "")
        #print("==question==",question,computer_ans[question]['value'])

        return f"{computer_ans[question]['value']}"
    elif (question == "data_collected_purpose" or question == "data_shared_purpose"):
        question = question.replace("_purpose", "")
        return f"{computer_ans[question]['value']}"


def sample_magic(magic, computer_data):
    magic_probabilities = [computer_data[question]['probability'] for question in magic]
    sum_magic_probabilities = sum(magic_probabilities)
    magic_c_data = {}

    if sum_magic_probabilities != 0:
        normalized_magic_probabilities = [prob / sum_magic_probabilities for prob in magic_probabilities]
        magic_size = len(magic)
        sampled_magic = np.random.choice(magic, magic_size, p=normalized_magic_probabilities, replace=False)

        for question in sampled_magic:
            if question in computer_data:
                magic_c_data[question] = computer_data[question]['value']

    return magic_c_data


def save_magic_data(magic_c_data):
    with open('magic-c.json', 'w') as magic_c_json_file:
        json.dump(magic_c_data, magic_c_json_file, indent=4)
