"""Find similar items application"""


import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader


load_dotenv()

openai_embd = OpenAIEmbeddings(model="text-embedding-ada-002")

def load_csv_file(csv_file):
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


def process_file(uploaded_file):
    """Save, load and create vectorstore
    """
    faiss_vector_store = None
    if uploaded_file is not None:
        temp_file_path = save_uploaded_file(uploaded_file)
        file_data = load_csv_file(temp_file_path)
        faiss_vector_store = create_faiss_vectorstore(file_data)
    return faiss_vector_store


def search(user_query, top_k=3):
    """Similarity search
    """
    results = faiss_vector_store.similarity_search_with_relevance_scores(user_query, k=top_k+10)
    results = list(filter(lambda x: x[1] > 0.7, results))
    search_results = []
    for res, score in results[:top_k]:
        search_results.append({"item": res.page_content.split(":")[0], "score": score})
    return search_results


st.set_page_config(page_title="Find Similar Item with LLM Embeddings")
st.header("Find Similar Items with LLM Embeddings")

uploaded_file = st.file_uploader("Upload CSV: ", type=["csv"], label_visibility="hidden")

faiss_vector_store = process_file(uploaded_file)

if faiss_vector_store is not None:
    st.subheader("Search similar things for: ")
    user_query = st.text_input("User Query: ", key="user_query", label_visibility="hidden")
    submit = st.button("Search")
    if submit:
        results = search(user_query)
        st.subheader("Similar Items: ")
        for res in results:
            st.write(f'{res["item"]} - {res["score"]}')