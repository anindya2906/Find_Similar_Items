"""Find similar items application"""


import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader


load_dotenv()

openai_embd = OpenAIEmbeddings(model="text-embedding-ada-002")

def upload_csv_file(csv_file):
    """Load csv data
    """
    loader = CSVLoader(file_path=csv_file, csv_args={
        "delimiter": ",",
        "quotechar": '"',
        "fieldnames": ["Items"]
    })
    data = loader.load()
    return data


def save_uploaded_file(file):
    """Save uploaded file
    """
    file_bytes_data = file.getvalue()
    temp_file_path = f"./example_data/{file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(file_bytes_data)
    return temp_file_path


def create_faiss_vectorstore(data):
    """Create FAISS index
    """
    faiss_vector_store = FAISS.from_documents(data, openai_embd)
    return faiss_vector_store


st.set_page_config(page_title="Find Similar Item with LLM Embeddings")
st.header("Find Similar Items with LLM Embeddings")

uploaded_file = st.file_uploader("Upload CSV: ", type=["csv"])

if uploaded_file is not None:
    temp_file_path = save_uploaded_file(uploaded_file)
    file_data = upload_csv_file(temp_file_path)
    faiss_vector_store = create_faiss_vectorstore(file_data)
    res = faiss_vector_store.similarity_search("Item2")
    for r in res:
        st.write(r.page_content)