import argparse
import json
import numpy as np
import os
import human
import computer
import sentence
import random
from string import Template 
import pandas as pd
import time
from create_json import create_human_response_json
import recommendation
import dataset


template_data = sentence.load_template()

def load_common_template():
    with open('json/common.json', 'r') as json_file:
        return json.load(json_file)

def get_history(reviewId,app_package,review):
    review = dataset.load_review_data()
    history = dataset.load_history()
    app_data = dataset.load_app_metadata() 
    

    df_filter=review[review['reviewId'] == reviewId]
    
    uid = df_filter["uid"].iloc[0]
    
    date = df_filter["formated_date"].iloc[0]
    
    date = pd.to_datetime(date)
    history_filter = history[history['uid'] == uid]
    
    df2_data = pd.merge(history_filter, review[['app_package','app_name']], on='app_package', how="left")
    df2_data = df2_data.drop_duplicates(subset='app_package')
    
    df2_data['formated_date'] = pd.to_datetime(df2_data['formated_date'])
    history_data = df2_data[df2_data['formated_date'] < date]

    return history_data

def simulator(computer_ask,human_response,reviewId,app_name,app_package,user_id,formated_date):

    # Sample questions from computer data
    human_a_data = human_response 
    computer_q_data = computer_ask    

    sampled_questions = computer.sample_questions(computer_q_data,reviewId)
    print("sampled_questions", sampled_questions)
    item_recommendation, negative_recommendations = recommendation.get_recommendations(reviewId,app_package)
    
    common_text = load_common_template()


    with open(f'conversations/mobileconvrec/conversations_{reviewId}.txt', 'w') as convo_file:
        st_time = time.time()
        convo_file.write(f"User Id: {user_id}\n\n")
        convo_file.write(f"User's Previous Interactions\n")
        history = get_history(reviewId,"app_package","review")
        
        if history.empty:
            convo_file.write("No History Available"f"\n")
        else:
            history_data = history.tail(10)            
                
            for index, row in history_data.iterrows():
                line = f"App Name: {row['app_name']} | Package Name: {row['app_package']} | Date: {row['formated_date']} | Rating: {row['rating']}\n"
                convo_file.write(line)
                
        convo_file.write(f"\n")
        
        recommended_app = f"Recommended App Name: "f"{app_name}"" | Package Name: "f"{app_package}"" | Date: "f"{formated_date}\n\n"
        
        convo_file.write(recommended_app)

        if len(negative_recommendations) == 0:
            convo_file.write("No Negative Recommendation Available"f"\n")
        else:
            for index, row in negative_recommendations.iterrows(): 
                negative_recommendation = f"Negative Recommended App Name: "f"{row['app_name']}"" | Package Name: "f"{row['app_package']}""\n"
                convo_file.write(negative_recommendation)

        
        end_time = time.time()
        elapsed_time = end_time - st_time
        print('Execution time:', elapsed_time, 'seconds')

        convo_file.write(f"\n")
        #ACtual Item Reco
        for index, row in item_recommendation.iterrows():
            positive_points = row['positive_points']
            pos_values = positive_points.split(", ")
            positive_pt_first = pos_values[0]
            positive_pt_rest = pos_values[1:]
            if len(positive_pt_rest) > 1:
                positive_pt_string = ", ".join(positive_pt_rest[:-1]) + " and " + positive_pt_rest[-1]
            else:
                positive_pt_string = positive_pt_rest[0]
        
        for question in sampled_questions:
            query_text = sentence.generate_sentence(question, template_data)
            query = random.choice(query_text['query'])

            human_val = human.human_answer(question, human_a_data)
            if human_val == '0':
                response = random.choice(query_text['text']["0"])
            elif human_val == '1':
                response = random.choice(query_text['text']["1"])
            else:
                response = random.choice(query_text['text']["1"])
            t_response = Template(response)
            
            computer_question = query
            if question == 'app_category':
                
                if 'app' or 'App' not in human_val:
                    human_val = human_val + " " + 'apps'
                else:
                    human_val = human_val
                common_computer_1 = random.choice(common_text['computer_1'])
                t_common_response_1 = Template(common_computer_1) 
                common_text_1 = t_common_response_1.substitute({'X': human_val})              

            human_response = t_response.substitute({'X': human_val})
            conversation_line = f"COMPUTER: {computer_question}\nHUMAN: {human_response}\n\n"
            convo_file.write(conversation_line)
            common_human_1 = random.choice(common_text['human_1'])
            t_common_human_1 = Template(common_human_1) 
            common_human_1 = t_common_human_1.substitute({'X': positive_pt_first})
            
            if question == 'app_category':               
                convo_file.write(f"COMPUTER: {common_text_1}\n")
                convo_file.write(f"HUMAN: {common_human_1}\n\n")
            
        #Negative Reco
        
        if len(negative_recommendations) == 0:
            print("No negative recommendation")
        else:
            neg = 0
            for index, row in negative_recommendations.iterrows():        
                positive_points = row['positive_points']
                negative_points = row['negative_points']
                key_word = row['key_word']
                neg_values = negative_points.split(", ")
                
                if len(neg_values) > 1:
                    neg_reco_pt_string = ", ".join(neg_values[:-1]) + " and " + neg_values[-1]
                else:
                    neg_reco_pt_string = neg_values[0]

                pos_values = positive_points.split(", ")
                pos_reco_pt_first = pos_values[0]
                pos_reco_pt_rest = pos_values[1:]
                if len(pos_reco_pt_rest) > 1:
                    pos_reco_ng_string = ", ".join(pos_reco_pt_rest[:-1]) + " and " + pos_reco_pt_rest[-1]
                elif len(pos_reco_pt_rest[0]) > 1:
                    pos_reco_ng_string = pos_reco_pt_rest[0]
                
                key_word_val = False
                if key_word != "None" and key_word.strip() != "":
                    key_word_val = True
                    q_word = key_word                
                    a_word = neg_values[0]
            
                #Neg Reco
                global computer_response_1, computer_response_2, computer_response_3, computer_response_4, computer_response_5, computer_response_6
                global human_response_1, human_response_2, human_response_3, human_response_4, human_response_5, human_response_6
                global t_user_response_2, t_user_response_4, t_user_response_6
                for i in range(1, 7):               
                    sys_key = f"computer_{i}"
                    user_key = f"human_{i}"
                    
                    globals()[f'sys_response_{i}'] = random.choice(common_text['negative_reco'][sys_key])
                    globals()[f'user_response_{i}'] = random.choice(common_text['negative_reco'][user_key])
                    
                    globals()[f't_sys_response_{i}'] = Template(globals()[f'sys_response_{i}'])
                    globals()[f't_user_response_{i}'] = Template(globals()[f'user_response_{i}'])                    
                    
                    if '$X' in globals()[f'sys_response_{i}']:
                        globals()[f'computer_response_{i}'] = globals()[f't_sys_response_{i}'].substitute({'X' : row['app_name']})
                    elif '$Y' in globals()[f'sys_response_{i}']:
                        globals()[f'computer_response_{i}'] = globals()[f't_sys_response_{i}'].substitute({'Y' : pos_reco_ng_string, 'Z': neg_reco_pt_string})
                    elif '$Z' in globals()[f'sys_response_{i}']:
                        globals()[f'computer_response_{i}'] = globals()[f't_sys_response_{i}'].substitute({'Z': neg_reco_pt_string})   
                    else:
                        globals()[f'computer_response_{i}'] = globals()[f'sys_response_{i}']
                    
                    
                    if '$X' in globals()[f'user_response_{i}']:
                        globals()[f'human_response_{i}'] = globals()[f't_user_response_{i}'].substitute({'X' : row['app_name']})
                    elif '$Y' in globals()[f'user_response_{i}']:
                        globals()[f'human_response_{i}'] = globals()[f't_user_response_{i}'].substitute({'Y' : pos_reco_pt_first})
                    elif '$Z' in globals()[f'sys_response_{i}']:
                        globals()[f'human_response_{i}'] = globals()[f't_user_response_{i}'].substitute({'Z': neg_reco_pt_string})
                    else:
                        globals()[f'human_response_{i}'] = globals()[f'user_response_{i}']
                    
                
                if key_word_val:
                    human_q = random.choice(common_text['negative_reco']['human_q'])
                    computer_a = random.choice(common_text['negative_reco']['computer_a'])
                    t_human_q = Template(human_q)
                    t_computer_a = Template(computer_a)
                    human_response_0 = t_human_q.substitute({'X' : q_word})
                    computer_response_0 = t_computer_a.substitute({'X' : a_word})             
                    human_response_2 = t_user_response_2.substitute({'Z' : a_word})
                    human_response_4 = t_user_response_4.substitute({'Z' : a_word})
                    human_response_6 = t_user_response_6.substitute({'Z' : a_word})

                if neg == 0:
                    convo_file.write(f"COMPUTER: {computer_response_1}")
                    
                    if key_word_val:
                        convo_file.write(f"\nHUMAN: {human_response_0}\n")
                        convo_file.write(f"\nCOMPUTER: {computer_response_0}")
                    else:
                        convo_file.write(f"\nHUMAN: {human_response_1}\n")
                        convo_file.write(f"\nCOMPUTER: {computer_response_2}")
                    convo_file.write(f"\nHUMAN: {human_response_2}\n")
                if neg == 1:
                    convo_file.write(f"\nCOMPUTER: {computer_response_3}")
                    if key_word_val:
                        convo_file.write(f"\nHUMAN: {human_response_0}\n")
                        convo_file.write(f"\nCOMPUTER: {computer_response_0}")
                    else:
                        convo_file.write(f"\nHUMAN: {human_response_3}\n")
                        convo_file.write(f"\nCOMPUTER: {computer_response_4}")
                    convo_file.write(f"\nHUMAN: {human_response_4}\n")
                if neg == 2:
                    convo_file.write(f"\nCOMPUTER: {computer_response_5}")
                    if key_word_val:
                        convo_file.write(f"\nHUMAN: {human_response_0}\n")
                        convo_file.write(f"\nCOMPUTER: {computer_response_0}")
                    else:
                        convo_file.write(f"\nHUMAN: {human_response_5}\n")
                        convo_file.write(f"\nCOMPUTER: {computer_response_6}")
                    convo_file.write(f"\nHUMAN: {human_response_6}\n")

                neg = neg + 1       
        
        #Computer Item
        computer_item_text = random.choice(common_text['computer_item'])
        t_computer_item = Template(computer_item_text)
        computer_item = t_computer_item.substitute({'X' : app_name})

        human_item_text = random.choice(common_text['human_item'])
        t_human_item = Template(human_item_text)
        human_item = t_human_item.substitute({'X' : app_name})
        #Computer End
        computer_end_text = random.choice(common_text['computer_end'])
        t_computer_end_text = Template(computer_end_text)
        computer_end = t_computer_end_text.substitute({'X' : app_name})
        #Human End
        human_end_text = random.choice(common_text['human_end'])
        t_human_end_text = Template(human_end_text)
        human_end = t_human_end_text.substitute({'X' : app_name})

        convo_file.write(f"\nCOMPUTER: {computer_item}")
        convo_file.write(f"\nHUMAN: {human_item}\n")
        
        convo_file.write(f"\nCOMPUTER: {app_name} app offers {positive_pt_string}.\n")
        convo_file.write(f"HUMAN: {human_end}\n")
        convo_file.write(f"\nCOMPUTER: {computer_end}\n")
        print("\nConversation ended & conversation saved in conversations.txt")
    return

def main(reviewId,app_name,app_package,user_id,formated_date):
    print("Review ID Mainsss====",reviewId)
    #create_computer_ask_json()
    create_human_response_json(reviewId) 
    
    # Sample questions from computer data
    computer_ask = computer.load_computer_data()
    human_response = human.load_human_data()

    simulator(computer_ask,human_response,reviewId,app_name,app_package,user_id,formated_date)

    return

def load_dataset(dataset_name):
    if dataset_name == 'mobileconvrec':
        review = dataset.load_review_data()
        id = 0
        error_ids=[]
    
        for index, row in review[review['reviewId'] >= id].iterrows():
            reviewId = row['reviewId']
            app_name = row['app_name']
            app_package = row['app_package']
            formated_date = row['formated_date']
            
            user_id= row['uid']
            try:
                print("Review Id:",reviewId,"for processing simulator.")
                print("\n")
                main(reviewId,app_name,app_package,user_id,formated_date)
                
            except ValueError as e:
                error_ids.append(reviewId)
                
    if dataset_name == 'tripadvisor':
        review = dataset.load_review_tripadvisor()
        print(review.columns)
        print(review.shape)
        
        error_ids=[]
    
        for index, row in review.iterrows():
        
            reviewId = row['id']
            #product_name = row['name'] #from Offering.txt
            product_id = row['offering_id']
            formated_date = row['date']
            
            user_id= row['author']['id']
            print(reviewId, product_id, formated_date, user_id)
            try:
                print("Review Id:",reviewId,"for processing simulator.")
                print("\n")
                #main(reviewId)
                
            except ValueError as e:
                error_ids.append(reviewId)       
    
    return

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--datafile', help='load Data file to simulate', required=False)
    
    args = vars(parser.parse_args())
    dataFile=args["datafile"]

    load_dataset(dataFile)


    

