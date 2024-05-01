import numpy as np
import json
import pandas as pd
import random
import dataset

def load_computer_data():
    with open('json/computer.json', 'r') as json_file:
        return json.load(json_file)


def sample_questions(computer_data, reviewId):
    # Sort keys based on probability
    sorted_keys = sorted(computer_data, key=lambda x: computer_data[x]["probability"], reverse=True)

    # Include the key with the highest probability at the beginning
    selected_keys = [sorted_keys[0]]

    # Randomly select 2 to 3 more keys based on their probabilities
    remaining_keys = random.sample(sorted_keys[1:], k=random.randint(2, min(3, len(sorted_keys) - 1)))

    # Combine selected keys with remaining keys
    selected_keys += remaining_keys
    return selected_keys