import streamlit as st
import os
import pandas as pd
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import time

# Load environment variables for disabling keys in frontend
load_dotenv()

# Load API keys for inferencing and embedding tool
os.environ['GROQ_API_KEY']= os.getenv("GROQ_API_KEY")
groq_api_key= os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN']= os.getenv("HF_TOKEN")

# Initialize embeddings and LLM
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Prompt
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only. The context is in structured format.
    Provide the most accurate response based on the question. 
    <context>
    {context}
    <context>
    Question: {input}
    """
)

def load_excel_files(directory="."):  # Default to current directory
    """Load all Excel files from the directory and convert them to text documents."""
    documents = []
    for file in os.listdir(directory):
        if file.endswith(".xlsx") or file.endswith(".xls"):
            file_path = os.path.join(directory, file)
            df = pd.read_excel(file_path)  # Load Excel dynamically
            text_data = df.to_string()  # Convert DataFrame to string
            documents.append(Document(page_content=text_data, metadata={"source": file}))
    return documents


def create_vector_embedding():
    """Process Excel files, split text, and create vector embeddings."""
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Load bank statements
        st.session_state.docs = load_excel_files(".")  # Using current directory
        
        # Split text into chunks
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        
        # Create FAISS vector store
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("Accounts Transactions Insights Bot")

user_prompt = st.text_input("Enter your query regarding the bank statements")

if st.button("Process Bank Statements"):
    create_vector_embedding()
    st.write("Vector Database is ready")

if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_prompt})
    st.write(f"Response Time: {time.process_time() - start} sec")

    st.write(response['answer'])

    ## Relevant document sections
    with st.expander("Relevant Transactions:"):
        for i, doc in enumerate(response['context']):
           st.write(doc.page_content)
           st.write('------------------------')
