from rag import RAG
from node import Node

if __name__ == "__main__":
    #If running for the first time uncomment the line below
    # rag = RAG(5, make_kg=True)

    #Ask Question
    rag = RAG(5)
    ans = rag.generate_answer('<Enter your Question>')
    #Example
    # ans = rag.generate_answer('What are the computational requirements?')
    


   