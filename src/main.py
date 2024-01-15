from components.vector_db import create_vector_db, connect_to_vector_db
from components.langchain_utils import create_chain
from components.langchain_helper import query_podcast
import box
import yaml
import os

def main():
    # Import config vars
    # with open('config.yaml', 'r', encoding='utf8') as ymlfile:
        # cfg = box.Box(yaml.safe_load(ymlfile))

    # Create/initialize client
    client = connect_to_vector_db()
    # Initialize chain
    rag_chain = create_chain(client)
    # Define query
    query = "Where was Kant born?"
    # Run chain
    output = rag_chain.invoke(query)
    print(output)
    

    '''
    # Retrieve context
    result = query_podcast(client, query)
    context = result["data"]["Get"]["Chatbot"]
    content = ' \n'.join(item["content"] for item in result["data"]["Get"]["Chatbot"])
    json_print(result)'''

    
    
if __name__ == "__main__":
    main()