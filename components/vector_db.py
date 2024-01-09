from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# TODO: add weaviate functionality
# TODO: check at start if db has already been created

def create_vector_db():
    # Load data
    loader = CSVLoader(file_path='data\podcasts.csv')
    data = loader.load()

    # Chunk text
    text_splitter = CharacterTextSplitter(chunk_size=25,
                                        chunk_overlap=5)
    chunked_documents = text_splitter.split_documents(data)

    # Load chunked documents into the FAISS index
    db = FAISS.from_documents(chunked_documents,
                            HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5'))

    retriever = db.as_retriever()

    return retriever