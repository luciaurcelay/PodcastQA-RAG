import json
import csv
from typing import Dict, List, Optional
from langchain.document_loaders.base import BaseLoader
from langchain.docstore.document import Document

class CustomCSVLoader(BaseLoader):
    """Loads a CSV file into a list of documents.

    Each document represents one row of the CSV file. Every row is converted into a
    key/value pair and outputted to a new line in the document's page_content.

    The source for each document loaded from csv is set to the value of the
    `file_path` argument for all doucments by default.
    You can override this by setting the `source_column` argument to the
    name of a column in the CSV file.
    The source of each document will then be set to the value of the column
    with the name specified in `source_column`.

    Output Example:
        .. code-block:: txt

            column1: value1
            column2: value2
            column3: value3
    """

    def __init__(
        self,
        file_path: str,
        source_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None,   # < ADDED
        csv_args: Optional[Dict] = None,
        encoding: Optional[str] = None,
    ):
        self.file_path = file_path
        self.source_column = source_column
        self.encoding = encoding
        self.csv_args = csv_args or {}
        self.metadata_columns = metadata_columns        # < ADDED

    def load(self) -> List[Document]:
        """Load data into document objects."""

        docs = []
        with open(self.file_path, newline="", encoding=self.encoding) as csvfile:
            csv_reader = csv.DictReader(csvfile, **self.csv_args)  # type: ignore
            for i, row in enumerate(csv_reader):
                content = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in row.items())
                try:
                    source = (
                        row[self.source_column]
                        if self.source_column is not None
                        else self.file_path
                    )
                except KeyError:
                    raise ValueError(
                        f"Source column '{self.source_column}' not found in CSV file."
                    )
                metadata = {"source": source, "row": i}
                # ADDED TO SAVE METADATA
                if self.metadata_columns:
                    for k, v in row.items():
                        if k in self.metadata_columns:
                            metadata[k] = v
                # END OF ADDED CODE
                doc = Document(page_content=content, metadata=metadata)
                docs.append(doc)

        return docs

def json_print(data):
    print(json.dumps(data, indent=2))

def dense_search(client, query, top_k=2):
    response = (
        client.query
        .get("Chatbot", ["content", "metadata"])
        .with_near_text({"concepts":[query]})
        .with_limit(top_k)
        .do()
    )
    return response


def sparse_search(client, query, top_k=3):
    response = (
        client.query
        .get("Chatbot",["content","metadata"])
        .with_bm25(query=query)
        .with_limit(top_k)
        .do()
    )
    return response


def hybrid_search(client, query, top_k=3):
    response = (
        client.query
        .get("Chatbot",["content","metadata"])
        .with_hybrid(query=query, alpha=0.5)
        .with_limit(top_k)
        .do()
    )
    return response