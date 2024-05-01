import numpy as np
import csv
import json
import pandas as pd
import random
import ast
import re
import dataset

#mobilerec
def create_human_response_json(reviewId):

    df_review = dataset.load_review_data()
        
    row = df_review[df_review['reviewId'] == reviewId]
    
    app_package = row['app_package'].iloc[0]    
    
    df = dataset.load_app_metadata()

    filtered_df = df[df['app_package'] == app_package]
    filtered_df = filtered_df[['app_category','app_type','content_rating','does_have_ads','num_reviews','price','avg_rating']]
    
    app_info_dict={}
    for column in filtered_df.columns:
        for index, row in filtered_df.iterrows():
            value = row[column]
            if column == 'app_category':
                val = row['app_type']
            elif column == 'price':
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
    
    human_dict = app_info_dict

    json_dict = json.dumps(human_dict, indent=4)
    
    with open('json/human.json', 'w') as jsonfile:
        json.dump(human_dict, jsonfile, indent=4)

