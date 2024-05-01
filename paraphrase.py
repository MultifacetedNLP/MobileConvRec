import openai
import time
import csv
import pandas as pd
import os
import re
import dataset

#load and set our key
openai.api_key = "" 

def custom_sort(file_name):
    return int(re.search(r'\d+', file_name).group())

def load_conv_txt():
    folder_path = 'conversations/mobileconvrec'

    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    sorted_files = sorted(files, key=custom_sort)
    
    for i, file_name in enumerate(sorted_files):
        file_path = os.path.join(folder_path, file_name)
        
        if file_name.startswith("conversations_") and file_name.endswith(".txt"):
        
            file_id = file_name.replace("conversations_", "").replace(".txt", "")
            revId = file_id
            
        print(f"Opening - {file_name}")
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            content = file.read()
            
            print(f"Content of {file_name}")
        
            split_content = content.split("COMPUTER:", 1)
            
            conversation_text = "COMPUTER:" + split_content[1].strip()    
            response = get_response(conversation_text)
            final_response = split_content[0].strip() + '\n\n' + response
            print(final_response)  
            #row.append(final_response)
            data = {
                "reviewId": revId,
                "response": final_response
            }

            df = pd.DataFrame(data, index=[i])
            df.to_csv('data/mobileconvrec/df_paraphrase_response.csv', mode='a', header=False)
    return 


def get_response(conversation_text):
    
    prompt = f"Here is the complete conversation text between model and user: \n"+ conversation_text + "\nRephrase the dialogues, keep a friendly tone, sentences short and use simple words, maintain the same format and response only."
    #print(prompt)
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", # this is "ChatGPT" $0.002 per 1k tokens
    messages=[{"role": "user", "content": prompt}]
    )
    print("reply_content===========")
    reply_content = completion.choices[0].message.content
                                 
    return reply_content


    
if __name__ == '__main__':
    load_conv_txt()
    