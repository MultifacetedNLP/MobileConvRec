import pandas as pd
import csv
import os
import dataset
import argparse

def extract_key_word(text, words):
    # Skip processing for NaN values
    if pd.isna(text):
        return 'None'

    # Split the text into two parts
    parts = text.split(', ')
    if len(parts) < 2:
        return 'None'
    
    part1_words = parts[0].lower().split()
    part2_words = parts[1].lower().split()

    # Check for words in both parts
    for word in words:
        if word.lower() in part1_words:
            return next((w for w in part1_words if w != word.lower()), 'None')
        elif word.lower() in part2_words:
            return next((w for w in part2_words if w != word.lower()), 'None')

    return 'None'

def get_key_word():
    # Load the DataFrame
    df = dataset.load_app_metadata

    # List of words to check for
    words = ["excessive","Inaccurate","Greedy","frustrating","Issues","Falls","claims","drains","poor","Difficult","cheating",
                "Unresponsive","Disruptive","Insufficient", "Limited","lags","Misleading","Unreliable","terrible","low","slow","problem",
                "expensive","Intrusive","Not","buggy","crashes","slows","fake","false","bad","unreliable","unbalanced","high","bugs",
                "lack","Annoying","disappointment","fault"]

    # Apply the function to create the 'key_word' column
    df['key_word'] = df['negative_points'].apply(lambda x: extract_key_word(x, words))

    df.to_csv('data/mobileconvrec/master_app_meta_data.csv')

def create_dialog():
    # Path to the CSV file
    csv_file_path = 'data/mobileconvrec/df_paraphrase_response.csv'

    # Output folder for text files
    output_folder = 'dialog'
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Read the CSV file and create a text file for each row
    with open(csv_file_path, 'r', newline='', encoding='utf-8', errors='ignore') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            # Extract 'id' and 'response' from the row
            dialog_id = int(row['reviewId'])
            response = row['response']
            
            # Create a unique text file name based on the 'id' column
            text_file_name = f"dialog_{dialog_id:05d}.txt"
            
            # Full path to the text file
            text_file_path = os.path.join(output_folder, text_file_name)
            
            # Write the 'response' to the text file
            with open(text_file_path, 'w', encoding='utf-8') as textfile:
                textfile.write(response)
                

    print(f"Text files have been created in the output folder: {output_folder}")

def process_dialog():
    folder_path = 'dialog'
    folder_path_out = 'dialogs'
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    for i, file_name in enumerate(files):
        file_path = os.path.join(folder_path, file_name)
        file_path_out = os.path.join(folder_path_out, file_name)
        
        if file_name.startswith("dialog_") and file_name.endswith(".txt"):
            with open(file_path, 'r', encoding='latin1') as file:
                content = file.read()
                print(f"Content of {file_name}")
                if 'AGENT:' in content or 'BOT:' in content or 'MODEL:' in content or 'Model:' in content or 'System:' in content or 'Computer:' in content or 'USER:' in content or 'Human:' in content or 'User:' in content:
                    content = content.replace('AGENT:', 'COMPUTER:')
                    content = content.replace('System:', 'COMPUTER:')
                    content = content.replace('Computer:', 'COMPUTER:')
                    content = content.replace('BOT:', 'COMPUTER:')
                    content = content.replace('MODEL:', 'COMPUTER:')
                    content = content.replace('Model:', 'COMPUTER:')

                    content = content.replace('USER:', 'HUMAN:')
                    content = content.replace('User:', 'HUMAN:')
                    content = content.replace('Human:', 'HUMAN:')
                
                else:
                    content = content
            with open(file_path_out, 'w', encoding='utf-8') as file:
                file.write(content)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Specify which function to run")

    parser.add_argument("function", choices=["create_dialog","get_key_word","process_dialog"], help="Function to run")
    args = parser.parse_args()
    
    if args.function == "get_key_word":
        get_key_word()
    if args.function == "create_dialog":
        create_dialog()
    elif args.function == "process_dialog":
        process_dialog()


