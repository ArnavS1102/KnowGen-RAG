import pandas as pd
import os
import re
from dotenv import load_dotenv
from gemini_api import make_prompt2, send_request
import json 

load_dotenv(".env")

def extract_dicts(json_string):
    try:
        data = json.loads(json_string)  
        if isinstance(data, list):
            return data  
        else:
            raise ValueError("JSON data is not a list.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return []

class Node:
    def __init__(self):
        self.prompt_path = os.getenv("JSON_FOLDER")
        self.csv_path  = os.getenv("CSV_FOLDER")

        with open(self.prompt_path, 'r') as file:
            data = json.load(file)

        self.sys_instruct = r"{}".format(data["Node"]["sys_instruct"])
        self.example_question1 = r"{}".format(data["Node"]["example_question1"])
        self.example_answer1 = r"{}".format(data["Node"]["example_answer1"])

    def replace_lines_in_csv(self,file_path):
    
        lines_to_remove = [
            r"The table contains three columns: Fruit, Color, and Taste, each representing a distinct characteristic of the listed fruits\.",
            r"The table lists four fruits: Apple, Banana, Lemon, and their corresponding colors and tastes\.",
            r"The colors represented are Red and Yellow, while the tastes are categorized as Sweet and Sour\.",
            r"The table implies that the taste of a fruit is independent of its color, as both Yellow fruits \(Banana and Lemon\) have different tastes \(Sweet and Sour, respectively\)\.",
            r"The table does not provide any information about the relationship between the fruit's color and taste, nor does it imply any causal link between the two\.",
            r"The classification system presented in the table is based on observable characteristics and does not involve any theoretical or abstract concepts\."
        ]
        
        patterns = [re.compile(line) for line in lines_to_remove]
        
        df = pd.read_csv(file_path)
        
        def remove_lines(text):
            if isinstance(text, str):
                for pattern in patterns:
                    text = pattern.sub('', text)  
            return text
        
        df = df.applymap(remove_lines)
        df.to_csv(file_path, index=False)

    def compute_nodes(self, str_):
        user_question = f"Now make a knowledge graph for this:\n{str_}"
        prompt = make_prompt2(self.sys_instruct, user_question, self.example_question1, self.example_answer1)
        response = send_request(prompt)
        li = extract_dicts(response.replace('json','').replace("```",""))
        dicts = [(temp_dict['node_1']['name'], temp_dict['node_2']['name'], temp_dict['relationship']) for temp_dict in li]
        str_values = ['({}, {}, {})'.format(node1, node2, relationship) for node1, node2, relationship in dicts]
       
        return str_values
    
    def add_nodes(self, file_path):
        df = pd.read_csv(file_path, usecols = ['text'])
        nodes = []
        for i in range(df.shape[0]):
            nodes_tup = self.compute_nodes(df.loc[i,"text"])
            nodes.append(nodes_tup)
        df['nodes'] = nodes
        df.to_csv(file_path)
        self.replace_lines_in_csv(self.csv_path)
    




        

    




    

