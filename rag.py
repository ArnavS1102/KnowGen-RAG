from get_kg import KG
from question_node import Question
from gemini_api import make_prompt1, send_request
import os
import json
from preprocess_pdf import PDF2MD
from postprocess_pdf import Cleaner, extract_last_itemize_block
from node import Node

class RAG:
    def __init__(self, no_docs, pre_clean = False):
        self.no_docs = no_docs
        self.prompt_path = os.getenv("JSON_FOLDER")
        self.source_dir = os.getenv("PDF_FOLDER")
        self.dest_dir = os.getenv("MD_FOLDER")
        self.csv_path = os.getenv("CSV_FOLDER")

        pdf2md = PDF2MD()
        cleaner = Cleaner()
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

        if pre_clean:
            pdf2md.parse_dir(self.dest_dir, self.source_dir)
            cleaner.clean_files(self.dest_dir)
        
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

