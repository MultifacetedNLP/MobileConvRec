import os
import json
import re
from datetime import datetime
import os
import re
from shutil import copyfile

def format_date(date_str):
    try:
        #parsing the date as YYYY-MM-DD HH:MM:SS
        date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            #parsing the date as DD-MM-YYYY
            date_obj = datetime.strptime(date_str, "%d-%m-%Y")
        except ValueError:
            
            return None

    # Format the date object as a string in the desired format
    formatted_date = date_obj.strftime("%Y-%m-%d")
    return formatted_date

def check_rec_in_message(rec, text):
    for r in rec:
        if r in text:
            return True
    return False

def partial_check_rec_in_message(rec, text):
    first_words = [name.split()[0] for name in rec]
    regex_pattern = r"\b(?:{})\b".format("|".join(re.escape(word) for word in first_words))
    match = re.search(regex_pattern, text, re.IGNORECASE)
    return bool(match)

def has_second_occurrence(lines, word):
    for line in lines:
        first_occurrence_index = line.find(word)
        if first_occurrence_index != -1:
            second_occurrence_index = line.find(word, first_occurrence_index + 1)
            if second_occurrence_index != -1:
                return True
    return False

def txt_to_jsonl():
    folder_path = 'dialogs'
    output_file = 'jsonl/MobileConvRec.jsonl'
    data_list = []
    
    # Loop through each file in the directory
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    for i, file_name in enumerate(files):
        file_path = os.path.join(folder_path, file_name)
        
        if file_name.startswith("dialog_") and file_name.endswith(".txt"):
            user_data = {
                'user_id':"",
                'user_previous_interactions': [],
                'recommended_app': {},
                'negative_recommended_app': [],
                'turns':[]
            }
            interaction_counter = 1
            current_turn = None
            reco = 0
            comp_interaction = 1
            computer_rec = []
            with open(file_path, 'r', encoding='latin1') as file:
                content = file.read()
            
                print(f"Content of {file_name}")
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                
                for line in lines:
                    user_match = re.match(r'User Id: (\w+)', line)
                    match_previous_interaction = re.match(r'App Name: (.+) \| Package Name: (.+) \| Date: (.+) \| Rating: (\d+)', line)
                    match_recommended = re.match(r'Recommended App Name: (.+) \| Package Name: (.+) \| Date: (.+)', line)
                    match_negative_recommended = re.match(r'Negative Recommended App Name: (.+) \| Package Name: (.+)', line)
                    match_interaction = re.match(r'(COMPUTER|HUMAN): (.+)', line)

                    if user_match:
                        user_id = user_match.group(1)
                        user_data['user_id'] = user_id
                    elif match_previous_interaction:
                        app_name, package_name, date, rating = match_previous_interaction.groups()
                        
                        app_name = re.sub(r'[^\x00-\x7F]+', '', app_name)
                        formatted_date = format_date(date)
                        interaction = {
                            'app_name': app_name,
                            'package_name':package_name,
                            'date':formatted_date,
                            'rating':rating
                        }
                        #f'App Name: {app_name} | Package Name: {package_name} | Date: {date} | Rating: {rating}'
                        user_data['user_previous_interactions'].append(interaction)

                    elif match_recommended:
                        rec_app_name, rec_package_name, rec_date = match_recommended.groups()
                        
                        rec_app_name = re.sub(r'[^\x00-\x7F]+', '', rec_app_name)
                        rec_formatted_date = format_date(date)
                        recommended_app = {
                            'app_name': rec_app_name,
                            'package_name':rec_package_name,
                            'date':rec_formatted_date
                        }
                        #f'Recommended App Name: {rec_app_name} | Package Name: {rec_package_name} | Date: {rec_date}'
                        user_data['recommended_app']= recommended_app
                        computer_rec.append(rec_app_name)                   
                    
                    elif match_negative_recommended:
                        app_name, package_name = match_negative_recommended.groups()
                        app_name = re.sub(r'[^\x00-\x7F]+', '', app_name)
                        negative_recommended_app ={
                            'app_name': app_name,
                            'package_name':package_name
                        }
                        user_data['negative_recommended_app'].append(negative_recommended_app)
                        computer_rec.append(app_name)
                        
                    elif match_interaction:

                        speaker, message = match_interaction.groups()
                        message = re.sub(r'[^\x00-\x7F]+', '', message)
                        user_accept = False
                        
                        if speaker == 'COMPUTER':
                            if check_rec_in_message(computer_rec, message) and reco != 1:
                                reco=True                                              
                            else:                                
                                reco=False
                                                               
                            current_turn = {'turn': interaction_counter, 'is_rec': reco, 'user_accept_recommendation': user_accept ,'COMPUTER': message}
                        elif speaker == 'HUMAN' and current_turn:
                            if "I'll give it a try" in message or "Thank you for the recommendation!" in message:
                                user_accept=True
                            elif check_rec_in_message([rec_app_name], message) and '?' not in message:
                                user_accept=True           
                            else:                               
                                if partial_check_rec_in_message([rec_app_name], message) and '?' not in message:
                                    user_accept=True
                                else:
                                    user_accept=False
                            current_turn['user_accept_recommendation'] = user_accept
                            current_turn['HUMAN'] = message
                            user_data['turns'].append(current_turn)
                            
                            current_turn = None
                        #user_data['turns'].setdefault(str(interaction_counter), []).append(f'{speaker}: {message}')
                        
                        if speaker == 'HUMAN':
                            interaction_counter += 1
                        if speaker == 'COMPUTER':
                            comp_interaction += 1
                        
               
                last_turn={
                    'turn':interaction_counter,
                    'is_rec':False,
                    'user_accept_recommendation': False,
                    'COMPUTER':message
                    
                }
                user_data['turns'].append(last_turn)
                data_list.append(user_data)
                
                

    # Write the data to a JSON Lines file
    with open(output_file, 'w', encoding='utf-8',) as jsonl_file:
        for data in data_list:
            jsonl_file.write(json.dumps(data) + '\n')
            print("created successfully")
    
    split_dataset()


def split_dataset():

    # Read the JSONL file and load its contents
    data = []
    with open('jsonl/MobileConvRec.jsonl', 'r') as file:
        for line in file:
            data.append(json.loads(line))

    # Sort the data based on the date
    data.sort(key=lambda x: datetime.strptime(x['recommended_app']['date'], '%Y-%m-%d'))

    # Calculate the number of records for each split
    total_records = len(data)
    test_size = int(0.2 * total_records)
    valid_size = int(0.1 * total_records)
    train_size = total_records - test_size - valid_size

    # Split the data into training, testing, and validation sets
    test_data = data[:test_size]
    valid_data = data[test_size:test_size+valid_size]
    train_data = data[test_size+valid_size:]

    # Write the split data to three separate JSONL files
    with open('jsonl/MobileConvRec_train.jsonl', 'w') as file:
        for record in train_data:
            file.write(json.dumps(record) + '\n')

    with open('jsonl/MobileConvRec_test.jsonl', 'w') as file:
        for record in test_data:
            file.write(json.dumps(record) + '\n')

    with open('jsonl/MobileConvRec_valid.jsonl', 'w') as file:
        for record in valid_data:
            file.write(json.dumps(record) + '\n')

if __name__ == '__main__':  
    txt_to_jsonl()
   