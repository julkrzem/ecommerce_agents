import os
import pandas as pd
from app.agents.vectorstore_retriver import VectorStore

df = pd.read_csv("app/data/sample.csv")

db = "./app/vectorstore/chroma_db"
columns = ['Clothing ID', 'Age', 'Rating', 'Recommended IND', 'Positive Feedback Count',
       'Division Name', 'Department Name', 'Class Name', 'Product_name']


vectorstore = VectorStore()

create_vectorstore = not os.path.exists(db)

documents, ids = vectorstore.prepare_documents(df, columns)
vectorstore.vector_store.add_documents(documents=documents, ids=ids)
print("db created")
