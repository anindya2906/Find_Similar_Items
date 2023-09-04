"""Find similar items application"""


import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader


load_dotenv()

def load_csv_data(csv_file):
    """Load csv data
    """
    loader = CSVLoader(file_path=csv_file, csv_args={
        "delimiter": ",",
        "quotechar": '"',
        "fieldnames": ["Items"]
    })
    data = loader.load()
    return data


st.set_page_config(page_title="Find Similar Item with LLM Embeddings")
st.header("Find Similar Items with LLM Embeddings")

uploaded_file = st.file_uploader("Upload CSV: ", type=["csv"])

if uploaded_file:
    load_csv_data(uploaded_file)