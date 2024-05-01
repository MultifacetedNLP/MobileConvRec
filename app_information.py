import openai
import time
import csv
import pandas as pd
import os
import re
import dataset

# load and set our key
openai.api_key = ""


def get_app_type_response(app_name,description,pos_reviews,neg_reviews):
    
    prompt = f"App Name: {app_name}\n\nHere is the apps description:\n{description}.\n\nHere is the list of positive reviews about the app:\n{pos_reviews}.\n\nHere is the list of negative reviews about the app:\n{neg_reviews}.\n\nPlease respond for below points,\n1. What does this app do (in 2-3 words)?\n2. Based on the app’s description and positive reviews, only list 3-4 positive points (no more than three words) about the app.\n3. Based on the app’s description and negative reviews, only list 1-2 negative points (no more than three words) about the app.\n\nPlease use the below format,\napp_type:\npositive_point:\nnegative_point:"
    #print(prompt)
    completion = openai.ChatCompletion.create(
    model= "gpt-3.5-turbo", # this is "ChatGPT" $0.002 per 1k tokens
    messages=[{"role": "user", "content": prompt}]
    )
    print("reply_content===========")
    reply_content = completion.choices[0].message.content
     
                            
    return reply_content

def get_positive_reviews(app_package):
    
    df_review = dataset.load_history()
    pos_review = []
    df_filter=df_review.loc[(df_review['app_package'] == app_package) & (df_review["rating"] == 5)]
    max_iterations = 5
    iteration_count = 0
    for index, row in df_filter.iterrows():
        if iteration_count >= max_iterations:
            break
        review = row["review"]
        pos_review.append(review)
        iteration_count += 1

    return pos_review

def get_negative_reviews(app_package):
    
    df_review = dataset.load_history()
    neg_review = []
    df_filter=df_review.loc[(df_review['app_package'] == app_package) & (df_review["rating"] == 1)]
    max_iterations = 5
    iteration_count = 0
    for index, row in df_filter.iterrows():
        if iteration_count >= max_iterations:
            break
        review = row["review"]
        neg_review.append(review)
        iteration_count += 1

    return neg_review

# Define a function to extract values
def extract_values(response):
    # Initialize variables to store extracted values
    app_type = None
    positive_points = []
    negative_points = []

    # Use regex to find matches
    app_type_match = re.search(r'app_type:\s*(.*)', response)
    
    if app_type_match:
        app_type = app_type_match.group(1).strip()
    
    if "1. " in response:
        
        lines = response.split("\n")
        for line in lines:
            if line.startswith("app_type:"):
                app_type = line.split(":")[1].strip()
            elif line.startswith("positive_point:"):
                positive_points.extend([point.strip() for point in lines[lines.index(line)+1:lines.index("negative_point:")]])
            elif line.startswith("negative_point:"):
                negative_points.extend([point.strip() for point in lines[lines.index(line)+1:]])
    elif "positive_point_" in response:
        
        lines = response.split("\n")
        for line in lines:
            if line.startswith("app_type:"):
                app_type = line.split(":")[1].strip()
            elif line.startswith("positive_point_"):
                positive_points.append(line.split(":")[1].strip())
            elif line.startswith("negative_point_"):
                points = line.split(":")[1].strip()
                
                if points:
                    negative_points.append(points)
                else:
                    negative_points.append("None")
    else:
        
        positive_point_matches = re.findall(r'positive_point:\s*(.*)', response)
        for match in positive_point_matches:
            positive_points.append(match.strip())

        negative_point_matches = re.findall(r'negative_point:\s*(.*)', response)
        for match in negative_point_matches:
            negative_points.append(match.strip())
    
    print(app_type, positive_points, negative_points)
    return app_type, positive_points, negative_points

#Update App Meta Data File to populate app_type, positive, negative
def get_app_type():
    df_app_meta = dataset.load_app_metadata()
    
    for index, row in df_app_meta.iterrows():
        
        app_name = row['app_name']
        desc = row['description']
        if len(desc.split()) > 305:
            desc_text = ' '.join(desc.split()[:300])
        else:
            desc_text = desc
        
        positive_review = get_positive_reviews(row['app_package'])
        negative_review = get_negative_reviews(row['app_package'])
        time.sleep(2)
        response = get_app_type_response(app_name, desc_text, positive_review,negative_review)
        #Extract app_type, positive, negative
        app_type, positive_feature, negative_feature= extract_values(response)
       
        positive_points = ', '.join(positive_feature)
        negative_points = ', '.join(negative_feature)
        
        data = {
                "app_package": row['app_package'],
                "app_name": row['app_name'],
                "app_type": app_type,
                "developer_name": row['developer_name'],
                "app_category": row['app_category'],
                "description": row['description'],
                "content_rating": row['content_rating'],
                "num_reviews": row['num_reviews'],
                "price": row['price'],
                "avg_rating": row['avg_rating'],
                "security_practices": row['security_practices'],
                "does_have_ads": row['does_have_ads'],
                "permission": row['permission'],
                "data_shared": row['data_shared'],
                "data_collected": row['data_collected'],
                "positive_points": positive_points,
                "negative_points": negative_points,
            }

        df = pd.DataFrame(data, index=[index])
        
        df.to_csv('data/master_app_meta_data.csv', mode='a', header=False)
    
if __name__ == '__main__':
    get_app_type()