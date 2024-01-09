from components.vector_db import create_vector_db
from langchain_helper import query_podcast


if __name__ == "__main__":
    # Initialize vector database
    retriever = create_vector_db()
    print(retriever)
    
    # Query retriever
    query = "What is the name of the egg recipe?"
    context = query_podcast(query, retriever)
    print(context)
