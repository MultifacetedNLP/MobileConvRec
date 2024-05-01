
-----------------------------------------------
Pre-Processing part for Dataset
-----------------------------------------------

STEP 1: Update [dataset.py] - To loading any other datasets 

STEP 2: Populate App MetaData File [app type, positive points, negative points]

REQUIRED: ChatGPT Key
RUN: python app_information.py - Add new column 'app type', 'positive points', 'negative points' in [dataset/mobileconvrec/master_app_meta_data.csv]

STEP 3: Populate key_words
RUN: python process_data.py get_key_word - Add new column 'key_word' in [dataset/mobileconvrec/master_app_meta_data.csv]

STEP 4: UPDATE STATIC TEMPLATES for Sentence Construction
- [json/computer_human_template.json] - Initial Question Answer
- [json/common.json] - Negative Recommendation and Conversation End

STEP 5: UPDATE FILES for Other Datasets
- update [json/computer.json] - Add aspects with probability
- update [computer.py] - add functions to load computer.json
- update [create_json.py] --- add functions to create human.json
- update [human.py] --- add functions to load human.json
- update [sentence.py] --- add functions to load any templates for conversation

-------------------------------------------------
Generate Conversations and Conversational Dataset
-------------------------------------------------

STEP 6: SIMULATOR
RUN: python main.py -d mobileconvrec - saves conversation as text files in [coversations] 

STEP 7: PARAPHRASE
RUN: python praphrase.py - ChatGPT response in file [dataset/mobileconvrec/df_paraphrase_response.csv]

STEP 8: 
RUN: python process_data.py create_dialog - saves paraphrase conversations as text file in [dialog]
RUN: python process_data.py process_dialog - removes any junk and saves paraphrase conversations as text file in [dialogs]

STEP 8: CREATE JSONL
RUN: python create_jsonl.py - create jsonl files (training, validation and testing) [jsonl]
