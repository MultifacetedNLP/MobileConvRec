import pandas as pd
import json

def load_review_data():
    df_review = pd.read_csv('data/mobileconvrec/mobilerec_review_data.csv')
    return df_review

def load_history():
    df_history = pd.read_csv('data/mobileconvrec/final_master_app_data.csv', usecols=['app_package','rating','uid','formated_date'])
    return df_history

def load_app_metadata():
    df_metadata = pd.read_csv('data/mobileconvrec/master_app_meta_data.csv')
    return df_metadata

def load_review_tripadvisor():
    df_tripadvisor = pd.read_json('data/tripadvisor/filtered_tripAdvisor_review.json')
    
    return df_tripadvisor




def load_yelp():
    pass

def load_amazon():
    pass

def load_goodreads():
    pass