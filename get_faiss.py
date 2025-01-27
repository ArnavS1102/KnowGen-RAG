import faiss

def build_faiss_index(embeddings):
        embedding_dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(embeddings)
        return index

def find_top_similar_nodes_faiss(graph_nodes, node_embedding, index, top_n=50):
        node_embedding = node_embedding.cpu().numpy().astype('float32').reshape(1, -1)
        _,top_n_indices = index.search(node_embedding, top_n)
        return [graph_nodes[idx] for idx in top_n_indices[0]]


    