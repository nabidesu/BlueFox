import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import tempfile
import uuid

# Asmita: Initialize session state for persistence
def initialize_session_state():
    """
    Initializes Streamlit's session state to store persistent data across user interactions.
    
    This function checks for the existence of key session state variables and initializes
    them if they do not exist. The variables include:
    - chat_history: A list to store the conversation history (user queries and assistant responses).
    - vectorstore: A FAISS vector store for resume content embeddings.
    - qa_chain: A RetrievalQA chain for processing questions against the resume.
    
    Args:
        None
    
    Returns:
        None
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

# Sushant: Load resume from uploaded file
def load_resume(file):
    """
    Loads and parses a PDF resume file into a list of document objects.
    
    This function saves the uploaded PDF file to a temporary directory with a unique
    filename, then uses PyPDFLoader from LangChain to extract the text content as
    document objects.
    
    Args:
        file: A Streamlit UploadedFile object representing the PDF resume.
    
    Returns:
        list: A list of document objects containing the parsed resume content.
    """
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, f"resume_{uuid.uuid4()}.pdf")
    with open(temp_file_path, "wb") as f:
        f.write(file.getvalue())
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    return documents

# Arun: Split documents into chunks
def split_documents(documents):
    """
    Splits a list of document objects into smaller chunks for efficient processing.
    
    This function uses RecursiveCharacterTextSplitter from LangChain to divide the
    documents into chunks of approximately 1000 characters with a 200-character overlap
    to maintain context between chunks.
    
    Args:
        documents: A list of document objects to be split.
    
    Returns:
        list: A list of document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    docs = text_splitter.split_documents(documents)
    return docs

# Ness: Create FAISS vector store with embeddings
def create_vectorstore(documents):
    """
    Creates a FAISS vector store from document chunks using text embeddings.
    
    This function uses the HuggingFaceEmbeddings model ('all-MiniLM-L6-v2') to generate
    embeddings for the document chunks, then stores them in a FAISS vector store for
    efficient similarity search during question answering.
    
    Args:
        documents: A list of document chunks to be embedded and stored.
    
    Returns:
        FAISS: A FAISS vector store containing the document embeddings.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# Sushant: Initialize Ollama LLM
def initialize_llm():
    """
    Initializes the Ollama language model for question answering.
    
    This function sets up the 'llama3.1:8b' model from Ollama, which will be used to
    generate responses based on the resume content.
    
    Args:
        None
    
    Returns:
        Ollama: An initialized Ollama language model instance.
    """
    llm = Ollama(model="llama3.1:8b")
    return llm

# Barsha: Define prompt template
def get_prompt_template():
    """
    Defines a prompt template for the RetrievalQA chain.
    
    This function creates a PromptTemplate object with a predefined template that
    instructs the language model to act as a professional HR assistant, using the
    provided resume context to answer questions accurately and professionally.
    
    Args:
        None
    
    Returns:
        PromptTemplate: A LangChain PromptTemplate object configured for resume analysis.
    """
    prompt_template = """You are a professional HR assistant analyzing a resume. Use the following context from the resume to answer the question accurately. If the information is not available in the context, say so.

    Context: {context}

    Question: {question}

    Answer in a professional tone, providing specific details from the resume when possible:
    """
    return PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

# Arun: Create RetrievalQA chain
def create_qa_chain(llm, vectorstore, prompt):
    """
    Creates a RetrievalQA chain for answering questions based on the resume.
    
    This function combines the Ollama language model, FAISS vector store, and prompt
    template to create a RetrievalQA chain that retrieves relevant document chunks
    and generates answers using the language model.
    
    Args:
        llm: An initialized Ollama language model instance.
        vectorstore: A FAISS vector store containing resume embeddings.
        prompt: A PromptTemplate object for formatting queries.
    
    Returns:
        RetrievalQA: A LangChain RetrievalQA chain configured for question answering.
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# Combined function to process resume 
def process_resume(file):
    """
    Processes a PDF resume file to prepare it for question answering.
    
    This function orchestrates the resume processing pipeline by loading the resume,
    splitting it into chunks, creating a vector store, initializing the language model,
    and setting up the RetrievalQA chain.
    
    Args:
        file: A Streamlit UploadedFile object representing the PDF resume.
    
    Returns:
        tuple: A tuple containing the FAISS vector store and the RetrievalQA chain.
    """
    documents = load_resume(file)
    docs = split_documents(documents)
    vectorstore = create_vectorstore(docs)
    llm = initialize_llm() 
    prompt = get_prompt_template() 
    qa_chain = create_qa_chain(llm, vectorstore, prompt)
    return vectorstore, qa_chain


st.set_page_config(page_title="Resume Chatbot", layout="wide")
st.title("📄 Resume Analysis Chatbot")
st.write("Upload a resume (PDF) and ask questions about the candidate's qualifications, experience, or skills.")

initialize_session_state()

with st.sidebar:
    st.header("Upload Resume")
    uploaded_file = st.file_uploader("Choose a PDF resume", type="pdf")
    
    if uploaded_file is not None:
        with st.spinner("Processing resume..."):
            st.session_state.vectorstore, st.session_state.qa_chain = process_resume(uploaded_file)
        st.success("Resume processed successfully!")

if st.session_state.vectorstore is not None and st.session_state.qa_chain is not None:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask a question about the resume"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.spinner("Analyzing resume..."):
            result = st.session_state.qa_chain({"query": prompt})
            response = result["result"]
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            with st.chat_message("assistant"):
                st.markdown(response)
                with st.expander("Source Documents"):
                    for doc in result["source_documents"]:
                        st.write(f"**Page {doc.metadata.get('page', 'N/A')}**: {doc.page_content[:200]}...")
else:
    st.info("Please upload a resume to start the analysis.")


if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()