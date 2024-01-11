from components.vector_db import create_vector_db, connect_to_vector_db
from components.misc import json_print
from langchain_helper import query_podcast


if __name__ == "__main__":
    # Create vector database
    client = create_vector_db()

    # Initialize vector database
    # client = connect_to_vector_db()

    # Define query
    query = "What are the ingredients of Benedict Cumberbatch Eggs?"

    # Retrieve context
    result = query_podcast(client, query)
    context = result["data"]["Get"]["Chatbot"]
    content = ' \n'.join(item["content"] for item in result["data"]["Get"]["Chatbot"])
    json_print(result)
    
    
