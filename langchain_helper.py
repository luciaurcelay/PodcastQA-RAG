def query_podcast(query, retriever):
    # Retrieve relevant documents
    docs = retriever.invoke(query)

    # Pass context and query to LLM
    answer = query
    
    return docs