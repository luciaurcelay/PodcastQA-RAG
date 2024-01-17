from components.vector_db import create_vector_db, connect_to_vector_db, preprocess_data
from components.langchain_utils import create_chain
import box
import yaml
from os.path import join, dirname
from langchain.globals import set_debug

set_debug(True)

def main():
    # Import config vars
    yaml_path = join(dirname(dirname(__file__)), 'config.yaml')
    with open(yaml_path, 'r', encoding='utf8') as ymlfile:
        cfg = box.Box(yaml.safe_load(ymlfile))
    
    # client = create_vector_db(cfg.chunking.chunk_size, cfg.chunking.chunk_overlap)
    # Create/initialize client
    client = connect_to_vector_db()
    # Initialize chain
    rag_chain = create_chain(client, cfg.k_top_chunks, cfg.llm.type, cfg.llm.model, cfg.memory.window)
    # Define query
    query = "Which fishes are the seven fishes??"
    # Run chain
    output = rag_chain.run(query)
    print(output)
    query2 = "Which one of them is simmered in tomato sauce garlic and red pepper flakes?"
    output2 = rag_chain.run(query2)
    print(output2)
    '''context = output['context']
    question = output['question']
    answer = output['text']
    print(f"Question: {question} \n \n Answer: {answer} \n \n")
    for d in context:
        print(f"Show name: {d.metadata['metadata']}")
        print(f"Content: {d.page_content}")'''
        

    
if __name__ == "__main__":
    main()