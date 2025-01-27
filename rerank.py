import cohere
import os
from dotenv import load_dotenv

load_dotenv(".env")

def re_rank(question, indices, df, no_docs):
    top_results_text = []
    relevant_texts = [df.loc[i, "text"] for i in indices]
    
    cohere_api_key = os.getenv("COHERE_API_KEY")
    co = cohere.Client(cohere_api_key)

    cohere_rerank_results = co.rerank(
              model='rerank-english-v2.0',
              query=question,
              documents=relevant_texts
          )
    top_idx_results = cohere_rerank_results.results[:min(no_docs, len(cohere_rerank_results.results))]
    
    for result in top_idx_results:
        doc_idx = result.index
        score = result.relevance_score
        doc_text = df.loc[indices[doc_idx], "text"] + '\n'

        top_results_text.append(f"{doc_text}")
    final_context = [f'<start_text>\n{result}\n<end_text>' for result in top_results_text]
    return final_context
    
    

