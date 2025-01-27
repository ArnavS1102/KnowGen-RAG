
from langchain.text_splitter import MarkdownHeaderTextSplitter
import os

class Splitter:
    def __init__(self):
        self.headers_to_split_on = [
            ("<start_doc>", "Document Name"),
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
            ("######", "Header 6"),
            ("#######", "Header 7"),
            ("########", "Header 8"),
            ("#########", "Header 9"),
            ("##########", "Header 10"),
        ]
        self.md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.headers_to_split_on)

    def ft_docs(self, str_):
        return self.md_splitter.split_text(str_)

    def get_list_ft(self, docs):
        li = []
        for doc in docs:
            if doc.metadata:
                chunk_str = f"\n<start_text>\n{doc.page_content}\n<end_text>"
                li.append(chunk_str)
        return li

    def get_dataset(self, docs):
        li = self.get_list_ft(docs)
        return li

    def ft_generate_splits(self, input_dir):
        splits = []
        for filename in os.listdir(input_dir):
            if filename.endswith(".md"):
                file_path = os.path.join(input_dir, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
                docs = self.ft_docs(text)
                li = self.get_dataset(docs)
                splits.extend(li)
        return splits