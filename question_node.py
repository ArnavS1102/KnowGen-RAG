import json
import re
import os

from gemini_api import make_prompt2, send_request
from dotenv import load_dotenv
from get_nodes import extract_dicts

load_dotenv(".env")

class Question:
    def __init__(self, q):
        self.prompt_path = os.getenv("JSON_FOLDER")

        with open(self.prompt_path, 'r') as file:
            data = json.load(file)

        self.sys_instruct_mqa = r"{}".format(data["Multi_QA"]["sys_instruct"])
        self.example_in_ques1 = r"{}".format(data["Multi_QA"]["example_in_ques1"])
        self.example_out_ques1 = r"{}".format(data["Multi_QA"]["example_out_ques1"])
        self.example_in_ques2 = r"{}".format(data["Multi_QA"]["example_in_ques2"])
        self.example_out_ques2 = r"{}".format(data["Multi_QA"]["example_out_ques2"])

        self.sys_instruct_node = r"{}".format(data["QA_NODE"]["sys_instruct"])
        self.example_in_node1 = r"{}".format(data["QA_NODE"]["example_in_node1"])
        self.example_out_node1 = r"{}".format(data["QA_NODE"]["example_out_node1"])
        self.example_in_node2 = r"{}".format(data["QA_NODE"]["example_in_node2"])
        self.example_out_node2 = r"{}".format(data["QA_NODE"]["example_out_node2"])
        
        self.q = q
        self.questions = self.get_qs()
        self.nodes1, self.nodes2 = self.get_nodes()

    def extract_items(self, text):
            itemize_blocks = re.findall(r'\\begin{itemize}(.*?)\\end{itemize}', text, re.DOTALL)
            if not itemize_blocks:
                return [] 
            last_block = itemize_blocks[-1]
            items = re.findall(r'\\item\s+(.*)', last_block)
            return items
    
    def get_qs(self):
        prompt = make_prompt2(self.sys_instruct_mqa, self.q, self.example_in_ques1, self.example_out_ques1, self.example_in_ques2, self.example_out_ques2)
        str_ = send_request(prompt)
        qs = self.extract_items(str_)
        return qs   

    def get_nodes(self):
        nodes1 = set()
        nodes2 = set()
        for q in self.questions:
            prompt = make_prompt2(self.sys_instruct_node, q, self.example_in_node1, self.example_out_node1, self.example_in_node2, self.example_out_node2)
            response = send_request(prompt)
            li = extract_dicts(response.replace('json','').replace("```",""))
            li1 = [temp_dict['node_1']['name'] for temp_dict in li]
            li2 = [temp_dict['node_2']['name'] for temp_dict in li]

            for i in li1:
                nodes1.add(i)
            for i in li2:
                nodes2.add(i)
        return nodes1, nodes2
    
     
    
    

    
    


