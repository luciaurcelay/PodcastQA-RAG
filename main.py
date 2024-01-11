from components.vector_db import create_vector_db
from components.misc import json_print
from langchain_helper import query_podcast


if __name__ == "__main__":
    # Initialize vector database
    client = create_vector_db()

    response = (
        client.query
        .get("Chatbot",["content","metadata"])
        .with_bm25(query="Where is Kant from?")
        .with_limit(3)
        .do()
    )

    json_print(response)
    
    # Query retriever
    query = "What is the name of the egg recipe?"
    # context = vector_db.similarity_search(query, top_k=3)
    # print(context)
    # answer = query_podcast(query, vector_db)
    
