from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Weaviate
import weaviate
import os
from os.path import join, dirname
from dotenv import load_dotenv
from . import utils

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


def create_vector_db(chunk_size, chunk_overlap):
    # Preprocess data
    data = preprocess_data(chunk_size, chunk_overlap)
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
                        "name": "show_name",
                        "dataType": ["text"],
                        "description": "Name of the show",
                        "moduleConfig": {
                            "text2vec-huggingface": {
                                "skip": True,
                                "vectorizePropertyName": False,
                            },
                        }
                    },
                    {
                        "name": "episode_name",
                        "dataType": ["text"],
                        "description": "Name of the episode",
                        "moduleConfig": {
                            "text2vec-huggingface": {
                                "skip": True,
                                "vectorizePropertyName": False,
                            },
                        }
                    },
                    {
                        "name": "row",
                        "dataType": ["int"],
                        "description": "Row of the csv",
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
                "show_name": d.metadata['show_name'],
                "episode_name": d.metadata['episode_name'],
                "row": d.metadata['row'],
            }
            batch.add_data_object(
                data_object=properties,
                class_name="Chatbot"
            )
    return client


def preprocess_data(chunk_size, chunk_overlap):
    # Load data
    loader = utils.CustomCSVLoader(
        file_path='src\data\podcasts.csv',
        # source_column="episode_name",
        metadata_columns = ["show_name","episode_name"],
        csv_args={
            "delimiter": ","}
        )
    data = loader.load()

    # Chunk text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunked_documents = text_splitter.split_documents(data)

    return chunked_documents


def print_vectors(client):
    result = (client.query
          .get("Chatbot", ["content"])
          .with_additional("vector")
          .with_limit(1)
          .do())

    json_print(result)


def print_obj_count(client):
    count = client.query.aggregate("Chatbot").with_meta_count().do()
    utils.json_print(count)