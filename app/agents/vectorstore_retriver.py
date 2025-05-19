from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd


class VectorStore():
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model="snowflake-arctic-embed:33m")
        self.vector_store = Chroma(
            collection_name="ecommerce_reviews",
            embedding_function=self.embeddings,
            persist_directory="./app/vectorstore/chroma_db",
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
    def prepare_documents(self, df, columns):
        documents = []
        ids = []
        
        for idx, row in df.iterrows():
            
            metadata={}
            for col in columns:
                metadata[col.replace(" ","_").lower()] = row[col]

            document = Document(
                page_content = row["Review Text"] if str(row["Title"])=="nan" else str(row["Title"]) + " " + row["Review Text"],
                metadata = metadata,
                id=str(idx)
            )
            ids.append(str(idx))
            documents.append(document)
        return documents, ids