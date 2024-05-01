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
