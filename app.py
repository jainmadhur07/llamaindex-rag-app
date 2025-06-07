

import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
import tempfile

# Set page config
st.set_page_config(
    page_title="PDF Q&A System",
    page_icon="📚",
    layout="wide"
)

# Environment variables (you can also put these in Streamlit secrets)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["GOOGLE_API_KEY"] = ""

# Initialize session state
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'llm' not in st.session_state:
    st.session_state.llm = None

def process_pdf(uploaded_file):
    """Process uploaded PDF and create retriever"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Load PDF
        with st.spinner("Loading PDF..."):
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            st.success(f"Loaded {len(pages)} pages from PDF")
        
        # Split text
        with st.spinner("Processing text..."):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
            splits = text_splitter.split_documents(pages)
            st.success(f"Created {len(splits)} text chunks")
        
        # Create vector store
        with st.spinner("Creating vector database..."):
            vectorstore = Chroma.from_documents(
                documents=splits, 
                embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
            st.success("Vector database created successfully!")
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return retriever, llm
        
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None, None

def ask_question(question, retriever, llm):
    """Get answer for a question"""
    try:
        # Retrieve relevant documents
        docs = retriever.invoke(question)
        
        # Create context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Simple prompt template
        prompt = f"""
        Based on the following context from the physics textbook, answer the question clearly and accurately.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        
        # Get response from LLM
        response = llm.invoke(prompt)
        return response.content
        
    except Exception as e:
        return f"Error processing question: {e}"

# Main UI
st.title("📚 PDF Q&A System")
st.markdown("Upload a PDF and ask questions about its content!")

# Sidebar for PDF upload
with st.sidebar:
    st.header("📁 Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to analyze"
    )
    
    if uploaded_file is not None:
        if st.button("Process PDF", type="primary"):
            retriever, llm = process_pdf(uploaded_file)
            if retriever and llm:
                st.session_state.retriever = retriever
                st.session_state.llm = llm
                st.rerun()
    
    # Show status
    if st.session_state.retriever is not None:
        st.success("PDF processed! Ready to answer questions.")
    else:
        st.info("Please upload and process a PDF to start asking questions.")

# Main content area
if st.session_state.retriever is not None:
    # Question input
    st.header("💬 Ask a Question")
    
    question = st.text_input(
        "Your question:",
        placeholder="e.g., What is explained in chapter 2?",
        key="question_input"
    )
    
    if st.button("Get Answer", type="primary") and question.strip():
        with st.spinner("Finding answer..."):
            answer = ask_question(question, st.session_state.retriever, st.session_state.llm)
            
            # Display the answer
            st.markdown("### 🤖 Answer:")
            st.markdown(answer)

else:
    # Welcome message
    st.info("👋 Welcome! Please upload a PDF file in the sidebar to get started.")
    
    st.markdown("""
    ### How to use:
    1. Upload a PDF file using the sidebar
    2. Click "Process PDF" to analyze the document
    3. Ask questions about the content
    4. Get instant answers based on the document
    
    ### Example questions:
    - "What is the main topic of chapter 2?"
    - "Explain electric field"
    - "What are the key concepts covered?"
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit 🎈 and LangChain 🦜")