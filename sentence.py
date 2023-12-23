import json

#computer - human
def load_template():
    with open('json/computer_human_template.json', 'r') as json_file:       
        return json.load(json_file)

def generate_sentence(question, template_data):
    query_text = template_data[question]
    return query_text

#human - computer
def load_question_answer():
    with open('json/human_computer_template.json', 'r') as json_file:
        return json.load(json_file)

def generate_question_answer(question, template_sentence):
    query_text = template_sentence[question]
    return query_text