import numpy as np
import csv
import json
import pandas as pd
import random
import ast
import re

aspect_columns = ['User Interface Design', 'Navigation', 'Accessibility',
           'Customization', 'Functionality', 'Performance',
           'Responsiveness', 'Security', 'Privacy', 'Permissions',
           'Data Collection', 'Data Sharing', 'Updates', 'Customer support',
           'Developer', 'In-app purchases', 'Battery/power Drainage']
basic = {
        "app_category": {
            "probability": 1
        },
        "content_rating": {
            "probability": 1
        },
        "does_have_ads": {
            "probability": 1
        },
        "num_reviews": {
            "probability": 1
        },
        "price": {
            "probability": 1
        },
        "avg_rating": {
            "probability": 1
        }
    }

group_aspect = ["app_category","content_rating", "does_have_ads", "num_reviews", "price", "avg_rating","permission","data_collected","data_shared"]
#Computer - Human 

def create_computer_ask_json():
    df = pd.read_csv('data/final_review_data_updated.csv')

    result = {}
    
    for column in df.columns[2:22]:  
        count_non_zero =  (df[column] != 0).sum()        
        
        probability = count_non_zero/len(df)
        result[column] = {
            #'value': column,
            'probability': probability,
        }
              
    computer_dict = {**result, **basic}

    result_json = json.dumps(computer_dict, indent=4)
    
    with open('json/computer.json', 'w') as jsonfile:
        json.dump(computer_dict, jsonfile, indent=4)

def create_human_response_json(reviewId):

    df_review = pd.read_csv('data/final_review_data_updated.csv')
        
    row = df_review[df_review['reviewId'] == reviewId]
    
    app_package = row['app_package'].iloc[0]
    aspect_dict = {column: {'value': 1 if int(value) > 0 else 0}
                for column, value in row.items() if column in aspect_columns
    }
    
    
    df = pd.read_csv('data/master_app_data_updated.csv')
    filtered_df = df[df['app_package'] == app_package]
    filtered_df = filtered_df[['app_category', 'content_rating', 'does_have_ads', 'num_reviews', 'price', 'avg_rating']]
    
    app_info_dict={}
    for column in filtered_df.columns:
        for index, row in filtered_df.iterrows():
            value = row[column]
            if column == 'price':
                val = 'Free' if value == 'Install' else "Paid"
            elif column == 'num_reviews':
                value = int(value.replace(',', ''))
                
                if int(value) <= 1000:
                    val = "< 1K"
                elif 1000 < int(value) <= 10000:
                    val = "10K"
                elif 10000 <int(value) <= 100000:
                    val = "100K"
                elif 100000 < int(value) <= 1000000:
                    val = "1M"
                elif 1000000 < int(value) <= 10000000:
                    val = "10M"
                elif 10000000 < int(value) <= 100000000:
                    val = "100M"
                elif 100000000< int(value) <= 1000000000:
                    val = "1B"
                else:
                    val = "> 1B"
            elif column == 'avg_rating':    
                val = 'Low' if float(value) < 3.0 else value
            else:
                val = value

            app_info_dict[column] = {
                #'name': column,
                'value': val
            }
    
    human_dict = {**app_info_dict, **aspect_dict}

    json_dict = json.dumps(human_dict, indent=4)
    
    with open('json/human.json', 'w') as jsonfile:
        json.dump(human_dict, jsonfile, indent=4)

#Human-Computer     
def create_human_ask_json():
    human_ask_dict ={}

    random_probability = random.uniform(0, 1)
    print("random_probabilityrandom_probability",random_probability)

    if random_probability < 0.4:
        group = ["app_category","content_rating", "does_have_ads", "num_reviews", "price", "avg_rating","permission"]
        for key in group:
            human_ask_dict[key] = {
                'probability': random_probability
            }
    elif 0.4 <= random_probability <= 0.6:
        group = ["permission", "data_collected"]
        for key in group:
            human_ask_dict[key] = {
                'probability': random_probability
            }
    elif random_probability > 0.6:
        group = ["data_collected", "data_shared"]
        for key in group:
            human_ask_dict[key] = {
                'probability': random_probability
            }
 
    for a in group_aspect:
        if a not in group:
            human_ask_dict[a] = {
                'probability': random.uniform(0, random_probability)
            }
  
    with open('json/human_ask.json', 'w') as jsonfile:
        json.dump(human_ask_dict, jsonfile, indent=4)
    return

def process_data_shared(value):
    shared = []
    for line in value:
        shared.append(line)
    return shared

def process_data(value):
    
    data_dict = json.loads(value.replace("'", "\""))
    keys = list(data_dict.keys())
    
    details = {key: list(inner_dict.keys()) for key, inner_dict in data_dict.items()}

    # Extract purposes for 'purpose'
    purposes = {inner_key: inner_value for outer_value in data_dict.values() for inner_key, inner_value in outer_value.items()}

    # Construct the final JSON-like structure
    result = {
        "value": keys,
        "detail": details,
        "purpose": purposes
    }
    return result

def process_permission(value):
    permission_list = value.split("\n")
    permission_list = [item.strip() for item in permission_list[1:] if item.strip()]
    # Initialize the result dictionary
    result = {'value': [], 'detail': {}}

    current_key = ''
    for item in permission_list:
        if item[0].isupper():
            current_key = item
            result['value'].append(item)
            result['detail'][item] = []
        else:
            result['detail'][current_key].append(item)
    return result


def create_computer_response_json(reviewId):
    computer2_dict = {}
    df_review = pd.read_csv('data/final_review_data_updated.csv')
    
        
    review_row = df_review[df_review['reviewId'] == reviewId]
    review_app_package = review_row['app_package'].iloc[0]

    df = pd.read_csv('data/master_app_data_updated.csv')
    
    app_row = df[df["app_package"]==review_app_package]
    selected_col = app_row[group_aspect]

    for column in selected_col.columns:
        for index, row in app_row.iterrows():
            value = row[column]
            
            if column == 'price':
                val = 'Free' if value == 'Install' else "Paid"
            elif column == 'num_reviews':
                if "," in str(value):
                    value = int(value.replace(',', ''))
                else:
                    value = value

                if int(value) <= 1000:
                    val = "< 1K"
                elif 1000 < int(value) <= 10000:
                    val = "10K"
                elif 10000 <int(value) <= 100000:
                    val = "100K"
                elif 100000 < int(value) <= 1000000:
                    val = "1M"
                elif 1000000 < int(value) <= 10000000:
                    val = "10M"
                elif 10000000 < int(value) <= 100000000:
                    val = "100M"
                elif 100000000< int(value) <= 1000000000:
                    val = "1B"
                else:
                    val = "> 1B"
            elif column == 'avg_rating':    
                val = 'Low' if float(value) < 3.0 else value
            elif column == 'data_shared':                
                val = process_data(value)
            elif column == 'data_collected':
                val = process_data(value)
                
            elif column == 'permission':
                val = process_permission(value)
            else:
                val = value

            computer2_dict[column] = {
                'value': val
            }
    
    with open('json/computer_response.json', 'w') as jsonfile:
        json.dump(computer2_dict, jsonfile, indent=4)
    return
