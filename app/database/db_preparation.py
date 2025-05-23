import os
import pandas as pd
from app.agents.rag_agent import RagAgent
from langchain_core.documents import Document

df = pd.read_csv("app/data/sample.csv")

db = "./app/database/chroma_db"
columns = ['Comment_id','Clothing ID', 'Age', 'Rating', 'Recommended IND', 'Positive Feedback Count',
       'Division Name', 'Department Name', 'Class Name', 'Product_name']


def chunk_text(text, chunk_size=50):
    words = text.split()
    chunks = [
        ' '.join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]
    return chunks

def prepare_documents(df, columns):
       documents = []
       ids = []

       idx = 0
       for i, row in df.iterrows():
       
              metadata={}
              for col in columns:
                     metadata[col.replace(" ","_").lower()] = row[col]
              
              content = row["Review Text"] if str(row["Title"])=="nan" else str(row["Title"]) + " " + row["Review Text"]

              chunks = chunk_text(content)

              for chunk in chunks:
                     document = Document(
                            page_content = chunk,
                            metadata = metadata,
                            id=str(idx)
                     )
                     ids.append(str(idx))
                     documents.append(document)
                     idx += 1
       return documents, ids

database = RagAgent()

# create_database = not os.path.exists(db)

documents, ids = prepare_documents(df, columns)
print(documents)
database.vector_store.add_documents(documents=documents, ids=ids)
print("db created")
