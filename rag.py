from get_kg import KG
from question_node import Question
from gemini_api import make_prompt1, send_request
import os
import json
from postprocess_pdf import extract_last_itemize_block
from node import Node
from dotenv import load_dotenv

load_dotenv(".env")

class RAG:
    def __init__(self, no_docs, make_kg = False):
        self.no_docs = no_docs
        self.prompt_path = os.getenv("JSON_FOLDER")
        self.source_dir = os.getenv("PDF_FOLDER")
        self.dest_dir = os.getenv("MD_FOLDER")
        self.csv_path = os.getenv("CSV_FOLDER")

        node = Node()

        with open(self.prompt_path, 'r') as file:
            data = json.load(file)

        self.sys_instruct = r"{}".format(data["GENERATOR"]["sys_instruct"])
        self.example_context1 = r"{}".format(data["GENERATOR"]["example_context1"])
        self.example_question1 = r"{}".format(data["GENERATOR"]["example_question1"])
        self.example_answer1 = r"{}".format(data["GENERATOR"]["example_answer1"])
        self.example_context2 = r"{}".format(data["GENERATOR"]["example_context2"])
        self.example_question2 = r"{}".format(data["GENERATOR"]["example_question2"])
        self.example_answer2 = r"{}".format(data["GENERATOR"]["example_answer2"])

        if make_kg:
            node.add_nodes(self.csv_path)
        
        self.graph = KG()

    def generate_answer(self, question):
        Q = Question(question)
        nodes = list(Q.nodes1) + list(Q.nodes2)
        context = self.graph.get_context(nodes, question, self.no_docs)
        prompt = make_prompt1(self.sys_instruct, "\n".join(context), question, self.example_context1, self.example_question1, self.example_answer1, self.example_context2, self.example_question2, self.example_answer2)
        answer = send_request(prompt)
        items  = extract_last_itemize_block(answer)
        return items.strip()

