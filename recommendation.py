import numpy as np
import csv
import json
import pandas as pd
import random
import dataset

def get_recommendations(reviewId, app_package):
    
    df_app_metadata = dataset.load_app_metadata()
    df_review = dataset.load_review_data()
    df_app = df_app_metadata[df_app_metadata['app_package'] == app_package]
    df_item_app = df_app[['app_package','app_name','app_category','app_type','positive_points','negative_points','key_word']]
    

    type = df_app['app_type'].iloc[0]
    category = df_app['app_category'].iloc[0]
    
    type_words = type.split()
    pattern = '|'.join(type_words)
    
    df_apps = df_app_metadata[df_app_metadata['app_type'].str.contains(pattern, case=False, na=False) & (df_app_metadata['app_category'] == category)]
    
    df_negative_apps = []
    if len(df_apps) > 1:
        filtered_app_packages = df_apps['app_package']
        
        # Filter the review DataFrame based on app_package and rating
        filtered_review_df = df_review[(df_review['app_package'].isin(filtered_app_packages)) & (df_review['rating'] == 1)]
        
        filter_review_app_packages = filtered_review_df['app_package'].unique()
        filtered_df_apps = df_apps[df_apps['app_package'].isin(filter_review_app_packages)]
        
        filtered_df_apps = filtered_df_apps[['app_package','app_name','app_category','app_type','positive_points','negative_points','key_word']]
        filtered_df_apps_final = filtered_df_apps[filtered_df_apps['app_package'] != app_package]
        
        if 0 < len(filtered_df_apps_final) < 4:
            max_num = len(filtered_df_apps_final)
        elif len(filtered_df_apps_final) >=4 :
            max_num = 3
        else:
            max_num = 0
        if max_num > 0:
            num_apps = random.randint(1, max_num)
            
        
        if len(filtered_df_apps_final)==0:
            df_negative_apps = []
        else:
            df_negative_apps = filtered_df_apps_final.sample(n=num_apps)
        
    return df_item_app, df_negative_apps