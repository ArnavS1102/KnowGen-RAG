
# KnowGen-RAG: A Hybrid Retrieval-Augmented Generation Framework

**KnowGen-RAG**, a hybrid Retrieval-Augmented Generation (RAG) framework designed to process highly technical documents by integrating knowledge graphs—comprising entities and relationships—with Large Language Model (LLM)-based natural language generation. This innovative approach enables more effective comprehension and contextualization of complex information.

---

## Table of Contents

1. [Features](#features)  
2. [Benefits](#benefits)  
3. [Project Structure](#project-structure)  
4. [How to Use](#how-to-use)  


---


## Features

1. **Integration with Knowledge Graphs**  
   KnowGen-RAG incorporates knowledge graphs to represent entities and relationships, providing a structured foundation for retrieving and contextualizing information.

2. **Indexing and Enrichment of Documents**  
   The framework indexes PDF documents using [Nougat's OCR](https://github.com/facebookresearch/nougat), enriches the content with detailed explanations, and organizes the data into a graph-like structure. This enhances the retrieval process and allows for precise extraction of context to address complex queries.

3. **Advanced LLM Integration**  
   KnowGen-RAG supports seamless integration with advanced LLMs, such as **Gemini** and **Cohere**, to further improve the quality and accuracy of generated answers.


## Benefits

- Improved understanding of highly technical documents.
- Enhanced ability to answer complex queries with contextually relevant information.
- Increased flexibility and scalability with support for multiple LLMs.
- Reliable assistance for decision-making and problem-solving, backed by domain expertise.

By leveraging knowledge graphs and advanced natural language generation, KnowGen-RAG provides a robust solution for processing unstructured documents to access domain-specific knowledge and expertise, serving as a reliable assistant that offers valuable insights and support to stakeholders and operators.

---

## Project Structure

```plaintext
├── sample.py           # Entry point of the application
├── requirements.txt    # List of dependencies
├── prompts.json        # Prompts
├── rag.py              # Entry point
├── preprocess_pdf.py   # For PDF to MD using OCR and additional cleaning
├── postprocess_pdf.py  # For processing tables, ewuations and additonal Regex patterns
├── md_split.py         # For markdown splitting
├── node.py             # For entity-relationship extraction
├── get_kg.py           # For generating the graph-based structure for searching
└── get_faiss.py        # For generating FAISS index
└── rerank.py           # For Cohere's API call (Re-Ranking)
└── gemini_api.py       # For Gemini's API call (Answer Generation)
└── gemini_api.py       # For Gemini's API call (Answer Generation)
```

---

## How to Use 
### 1. Create Required Directories
Run the following command to create necessary directories:

```sh
python make_dirs.py
```

### 2. Upload PDF Files
Place all your PDF documents into the `./pdf` directory.

### 3. Run OCR on PDFs
Execute the following script to perform Optical Character Recognition (OCR) on the uploaded PDFs:

```sh
python ocr.py
```

### 4. Run the RAG-based System
Now, you can run the `sample.py` script.

#### First-time setup
Uncomment the following line in `sample.py` before running:

```python
rag = RAG(5, make_kg=True)
```
For subsequent runs
Use the following command to generate an answer by providing your question:

```python
ans = rag.generate_answer('<Enter your Question>')
```


### 5. Execute the Script
Run the script:

```sh
python sample.py
```




 


