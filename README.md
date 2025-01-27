
# KnowGen-RAG: A Hybrid Retrieval-Augmented Generation Framework

**KnowGen-RAG**, a hybrid Retrieval-Augmented Generation (RAG) framework designed to process highly technical documents by integrating knowledge graphs—comprising entities and relationships—with Large Language Model (LLM)-based natural language generation. This innovative approach enables more effective comprehension and contextualization of complex information.

---

## Table of Contents

1. [Features](#features)  
2. [Benefits](#benefits)  
3. [Project Structure](#project-structure)  
4. [Setup Instructions](#setup-instructions)  
5. [How to Use](#how-to-use)  
6. [Dependencies](#dependencies)  
7. [Contributing](#contributing)  
8. [License](#license)  


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


List the key features of the project:

- Feature 1
- Feature 2
- Feature 3

Example:
- Parses and processes large datasets
- Generates custom reports
- Provides an interactive user interface

---

## Project Structure

Provide a breakdown of the files and their purpose:

```plaintext
├── main.py             # Entry point of the application
├── utils.py            # Contains utility functions used across the project
├── config.py           # Configuration and settings
├── requirements.txt    # List of dependencies
├── README.md           # Documentation for the project
├── data/               # Folder for input data files
├── output/             # Folder for generated outputs
└── tests/              # Folder containing unit tests
