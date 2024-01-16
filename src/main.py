from components.vector_db import create_vector_db, connect_to_vector_db
from components.langchain_utils import create_chain
import box
import yaml
from os.path import join, dirname
from langchain.globals import set_debug

# set_debug(True)

def main():
    # Import config vars
    yaml_path = join(dirname(dirname(__file__)), 'config.yaml')
    with open(yaml_path, 'r', encoding='utf8') as ymlfile:
        cfg = box.Box(yaml.safe_load(ymlfile))
    
    # client = create_vector_db()
    # Create/initialize client
    client = connect_to_vector_db()
    # Initialize chain
    rag_chain = create_chain(client)
    # Define query
    query = "What are the ingredients of Benedict Cumberbatch Eggs?"
    # Run chain
    output = rag_chain.invoke(query)
    context = output['context']
    question = output['question']
    answer = output['text']
    print(f"Question: {question} \n \n Answer: {answer} \n \n")
    for d in context:
        print(f"Show name: {d.metadata['metadata']}")
        print(f"Content: {d.page_content}")
        

    
if __name__ == "__main__":
    main()