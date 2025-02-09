import os
import re
import time
import json
import zipfile

import numpy as np
import pandas as pd
import networkx as nx

import rerank

from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from gemini_api import make_prompt2, send_request
from get_faiss import build_faiss_index, find_top_similar_nodes_faiss
from rerank import re_rank

load_dotenv(".env")

class KG:
    def __init__(self):
        self.csv_path = os.getenv("CSV_FOLDER")
        self.prompt_path = os.getenv("JSON_FOLDER")
        self.df = pd.read_csv(self.csv_path, usecols = ['text', 'nodes'])
        dict = self.get_nodes(self.df)
        self.G = self.construct_graph(dict)
        self.graph_nodes = list(self.G.nodes)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2',trust_remote_code=True)
        self.graph_node_embeddings = self.get_embeddings()
        self.index = build_faiss_index(self.graph_node_embeddings)
        
    def get_nodes(self, df):
        pattern = r"<([^>]+)>"
        dict_ = {}
        li = []
        str_ = [i.replace("['","").replace("]'","") for i in df['nodes']]

        for i in range(len(str_)):
            matches = re.findall(pattern, str_[i])
            tuples_list = [tuple(map(str.strip, match.split(',', 2))) for match in matches]
            for tuple_ in tuples_list:
                dict_[tuple_] = i

        return dict_

    def construct_graph(self,dict_):
        G = nx.DiGraph()
        keys = [key for key in dict_.keys()]
        for key in keys:
            if len(key) == 3:
                node1 = key[0]
                node2 = key[1]
                relationship = key[2]
                G.add_edge(node1, node2, label=f'{relationship} DF_INDEX:{dict_[key]}')
        return G
    
    def get_embeddings(self, nodes = None):
        if not nodes:
            nodes = list(self.G.nodes)
            node_embeddings = self.embedding_model.encode(nodes, convert_to_tensor=True)
            return node_embeddings.cpu().numpy().astype('float32')
        else:
            node_embeddings = self.embedding_model.encode(nodes, convert_to_tensor=True)
            return node_embeddings.cpu().numpy().astype('float32')
    
    def get_indices(self, nodes):
        pattern = re.compile(r'DF_INDEX:(\d+)$')
        unique_indices = set()

        for node in nodes:
            neighbors = list(self.G.neighbors(node))
            for neighbor in neighbors:
                if self.G.has_edge(node, neighbor):
                    edge_label = self.G.edges[node, neighbor].get('label', '')
                    match = pattern.search(edge_label)
                    if match:
                        unique_indices.add(int(match.group(1)))
        return list(unique_indices)
    
    def get_context(self, nodes, question, no_docs):
        li = []

        for node in nodes:
            embd = self.embedding_model.encode(node, convert_to_tensor=True)
            x = find_top_similar_nodes_faiss(self.graph_nodes, embd, self.index)
            li.extend(x)

        li = [i for i in set(li)]
        
        indices = self.get_indices(li)
        context = re_rank(question, indices, self.df, no_docs)
        return context




    





        




        



