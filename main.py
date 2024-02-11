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
import copy
from create_json import create_computer_ask_json, create_human_response_json, create_human_ask_json, create_computer_response_json

template_data = sentence.load_template()
template_sentence = sentence.load_question_answer()

def load_common_template():
    with open('json/common.json', 'r') as json_file:
        return json.load(json_file)

def get_history(reviewId,app_package,review):

    df_filter=df1[df1['reviewId'] == reviewId]
    
    uid = df_filter["uid"].iloc[0]
    print(uid)
    date = df_filter["formated_date"].iloc[0]
    print(date)
    date = pd.to_datetime(date)
    df2_filter = df2[df2['uid'] == uid]
    
    df2_data = pd.merge(df2_filter, df1[['app_package','app_name']], on='app_package', how="left")
    df2_data = df2_data.drop_duplicates(subset='app_package')
    
    df2_data['formated_date'] = pd.to_datetime(df2_data['formated_date'])
    history_df = df2_data[df2_data['formated_date'] < date]

    return history_df

def simulator(computer_ask,human_response,computer_response,human_ask):

    # Sample questions from computer data
    human_a_data = human_response 
    computer_q_data = computer_ask    

    sampled_questions = computer.sample_questions(computer_q_data,reviewId)

    # Sample questions from human data
    human_q_data = human.load_human_question()
    computer_a_data = computer.load_computer_answer_data()    

    human_questions = human.human_question(human_q_data)
    common_text = load_common_template()
    
    with open(f'conversation/conversations_{reviewId}.txt', 'w') as convo_file:
        st_time = time.time()
        convo_file.write(f"User Id: {user_id}\n\n")
        
        history = get_history(reviewId,"app_package","review")
        
        if history.empty:
            convo_file.write("No History Available"f"\n\n")
        else:
            history_data = history.tail(10)       
            for index, row in history_data.iterrows():
                line = f"App Name: {row['app_name']} | Date: {row['formated_date']} | Rating: {row['rating']}\n"
                convo_file.write(line)
        
        convo_file.write(f"\n")

        convo_file.write(f"COMPUTER: {common_text['computer_1']}\n")
        convo_file.write(f"HUMAN: {common_text['human_1']}\n\n")
        
        end_time = time.time()
        elapsed_time = end_time - st_time
        print('Execution time:', elapsed_time, 'seconds')

        convo_file.write(f"\n")
        
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
            
            human_response = t_response.substitute({'X' : human_val})

            conversation_line = f"COMPUTER: {computer_question}\nHUMAN: {human_response}\n\n"
            convo_file.write(conversation_line)
        aap_name = app_name
        convo_file.write(f"\nCOMPUTER: I would like to recommend "f"{aap_name}"" app for you! " f"{common_text['computer_2']} \n")
        
        convo_file.write(f"HUMAN: {common_text['human_2']}\n")
        convo_file.write(f"COMPUTER: {common_text['computer_4']}\n")
        convo_file.write(f"\n")
    
        list_keys = ['data_collected','data_shared','permission']
        for item in list_keys:
            if item in human_questions:                   
                feature_count = human.process_app_feature(item)
                
                if item == "data_collected":
                    if (2 <= feature_count <= 3 ):
                        index = np.where(human_questions == item)[0][0] + 1
                        new_key = np.array([f'data_collected_detail'], dtype=object)
                        human_questions = np.concatenate((human_questions[:index], new_key, human_questions[index:]))
                        
                    elif (feature_count > 3 ):
                        index = np.where(human_questions == item)[0][0] + 1
                        new_key = np.array([f'data_collected_detail', f'data_collected_purpose'], dtype=object)
                        
                        human_questions = np.concatenate((human_questions[:index], new_key, human_questions[index:]))

                        
                elif item == "data_shared":
                    if (2 <= feature_count <= 3 ):
                        index = np.where(human_questions == item)[0][0] + 1
                        new_key = np.array([f'data_shared_detail'], dtype=object)
                        human_questions = np.concatenate((human_questions[:index], new_key, human_questions[index:]))

                        
                    elif (feature_count > 3 ):
                        index = np.where(human_questions == item)[0][0] + 1
                        new_key = np.array([f'data_shared_detail', f'data_shared_purpose'], dtype=object)
                        
                        human_questions = np.concatenate((human_questions[:index], new_key, human_questions[index:]))

                        
                elif item == "permission":
                    if (feature_count >= 2.5 ):
                        index = np.where(human_questions == item)[0][0] + 1
                        new_key = np.array([f'permission_detail'], dtype=object)
                        human_questions = np.concatenate((human_questions[:index], new_key, human_questions[index:]))

                else:
                    human_questions = human_questions
        
        for question in human_questions:
        
            question_text = sentence.generate_question_answer(question, template_sentence)
            query = random.choice(question_text['question'])

            computer_val = computer.computer_answer(question, computer_a_data)
            
            data_dict = copy.copy(computer_val)
            
            if question == "data_collected" or question == "data_shared" or question == "permission":
                computer_val_prev = 0
                data_dict = json.loads(computer_val.replace("'", "\""))
                
                if data_dict['value']:
                    selected_keys = random.sample(data_dict['value'], k=1)
                    computer_val = ', '.join(data_dict['value'])
                else:
                    computer_val = -1
                    computer_val_prev = -1
            elif question == "data_collected_detail" or question == "data_shared_detail" or question == "permission_detail":
                
                if computer_val_prev !=-1:
                    data_dict = json.loads(computer_val.replace("'", "\""))
                    
                    if data_dict['detail']:
                        detail_info = {key: data_dict['detail'][key] for key in selected_keys}
                        extracted_detail = []
                        for key, value in detail_info.items():
                            if value:
                                extracted_detail.append(f'{key} are {value[0]}')
                                t_query = Template(query)
                                human_query = t_query.substitute({'Y' : f'{key}'})
                                query = human_query
                        
                        result_detail = ', '.join(extracted_detail)
                        computer_val = result_detail 
                    else:
                        computer_val = -1
                else:
                    continue
            elif question == "data_collected_purpose" or question == "data_shared_purpose":
                if computer_val_prev !=-1:
                    data_dict = json.loads(computer_val.replace("'", "\""))
                    
                    if data_dict['purpose']:
                        random_detail_keys = random.sample(detail_info.keys(), k=1)
                    
                        selected_detail_info = [item for sublist in detail_info.values() for item in sublist]
                        purpose_info_keys = random.sample(selected_detail_info, k=1)
                    
                        purpose_info = {key: data_dict['purpose'][key] for key in purpose_info_keys}
                        
                        extracted_purpose = []
                        for key, value in purpose_info.items():
                            if value:
                                extracted_purpose.append(f'{key} is {value}')
                                t_query = Template(query)
                                human_query = t_query.substitute({'Y' : f'{key}'})
                                query = human_query
                    
                        result_purpose = ', '.join(extracted_purpose)
                        computer_val = result_purpose 
                                       
                    else:
                        computer_val = -1
                else:
                    continue
        
            response = random.choice(question_text['response'])
            t_response = Template(response)
            
            human_question = query

            if computer_val !=-1:
                computer_response = t_response.substitute({'X' : computer_val})
            else:
                if question == "data_collected":
                    computer_response = "No data collected by the app."
                elif question == "data_shared":
                    computer_response = "No data shared by the app." 
                elif question == "permission":
                    computer_response = "No permission is granted for the app."
            
            conversation_line = f"HUMAN: {human_question}\nCOMPUTER: {computer_response}\n\n"

            convo_file.write(conversation_line)

        convo_file.write(f"\n{common_text['computer_3']}\n")
        print("\nConversation ended & conversation saved in conversations.txt")
    return



def main(reviewId):
    
    #create_computer_ask_json()
    create_human_response_json(reviewId)
    create_human_ask_json() 
    create_computer_response_json(reviewId)

    # Sample questions from computer data
    human_response = human.load_human_data()
    computer_ask = computer.load_computer_data()

    # Sample questions from human data
    human_ask = human.load_human_question()
    computer_response = computer.load_computer_answer_data()

    simulator(computer_ask,human_response,computer_response,human_ask)
    return

if __name__ == '__main__':
    
    df1 = pd.read_csv('data/final_review_data_updated.csv')
    df2 = pd.read_csv('data/final_master_app_data.csv', usecols=['app_package','rating','uid','formated_date'])
    id=0

    for index, row in df1[df1['reviewId'] >= id].iterrows():
        reviewId = row['reviewId']
        app_name = row['app_name']
        user_id= row['uid']
        print("Review Id:",reviewId,"for processing simulator.")
        print("\n")
        main(reviewId)
    
    
