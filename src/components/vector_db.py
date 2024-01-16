from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Weaviate
import weaviate
from components.misc import json_print
import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = dotenv_path = join(dirname(dirname(dirname(__file__))), '.env')
load_dotenv(dotenv_path)
auth_config = weaviate.AuthApiKey(api_key=os.environ.get("WEAVIATE_API_KEY"))
WEAVIATE_URL = os.environ.get("WEAVIATE_URL")

def connect_to_vector_db():
    client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=auth_config
    )
    if client.is_ready():
        print("Client is ready")
    else:
        print("Could not connect to client")
    return client


def update_vector_db():
    pass


def create_vector_db():
    # Preprocess data
    data = preprocess_data()
    # Connect Weaviate Cluster
    client = weaviate.Client(
        url=WEAVIATE_URL,
        additional_headers={"X-HuggingFace-Api-Key": os.environ.get("HUGGINGFACEHUB_API_TOKEN")},
        auth_client_secret=auth_config
    )
    # Define schema
    if client.schema.exists("Chatbot"):
        client.schema.delete_class("Chatbot")
    schema = {
        "classes": [
            {
                "class": "Chatbot",
                "description": "Documents for podcast-chatbot",
                "vectorizer": "text2vec-huggingface",
                "moduleConfig": {
                    "text2vec-huggingface": {
                        "model": "BAAI/bge-base-en-v1.5",
                        "options": {
                            "waitForModel": True
                            }
                    }
                },
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "Content of the chunk",
                        "moduleConfig": {
                            "text2vec-huggingface": {
                                "skip": False,
                                "vectorizePropertyName": False,
                            },
                        }
                    },
                    {
                        "name": "metadata",
                        "dataType": ["text"],
                        "description": "Metadata of the chunk",
                        "moduleConfig": {
                            "text2vec-huggingface": {
                                "skip": True,
                                "vectorizePropertyName": False,
                            },
                        }
                    }
                ]
            }
        ]
    }

    # Create defined schema
    client.schema.create(schema)
    # Populate db with data
    with client.batch.configure(batch_size=5) as batch:
        for i, d in enumerate(data):  # Batch import data
            print(f"importing doc: {i+1}")
            properties = {
                "content": d.page_content,
                "metadata": d.metadata['source'],
            }
            batch.add_data_object(
                data_object=properties,
                class_name="Chatbot"
            )
    return client


def preprocess_data():
    # Load data
    loader = CSVLoader(
        file_path='src\data\podcasts.csv',
        source_column="episode_name",
        csv_args={
            "delimiter": ",",}
        )
    data = loader.load()

    # Chunk text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunked_documents = text_splitter.split_documents(data)

    return chunked_documents


def create_weaviate_db_instance(client, data):
    #Lload text into the vectorstore
    vector_db = Weaviate(client, "Chatbot", "content", attributes=["source"])

    # load text into the vectorstore
    text_meta_pair = [(doc.page_content, doc.metadata) for doc in data]
    texts, meta = list(zip(*text_meta_pair))
    vector_db.add_texts(texts, meta)

    # Query
    query = "what is a vectorstore?"
    # Retrieve text related to the query
    docs = vector_db.similarity_search(query, top_k=20)

    return vector_db


def print_vectors(client):
    result = (client.query
          .get("Chatbot", ["content"])
          .with_additional("vector")
          .with_limit(1)
          .do())

    json_print(result)


def print_obj_count(client):
    count = client.query.aggregate("Chatbot").with_meta_count().do()
    json_print(count)