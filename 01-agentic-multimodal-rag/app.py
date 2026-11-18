from pathlib import Path
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from tools import process_and_index_pdfs
from agents import create_multimodal_rag_graph

# 1. Page Configuration
st.set_page_config(page_title="Multimodal Agentic RAG", page_icon="🤖", layout="wide")
st.title("📄 MultiModal Agentic RAG Assistant")

# Project Directory Configuration
BASE_DIR = Path(__file__).resolve().parent

TEMP_DIR = BASE_DIR / "temp_uploads"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# 2. Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed" not in st.session_state:
    st.session_state.indexed = False

# 3. Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Provider & API Key Selection
    provider = st.selectbox("Choose API Provider", ["Google Gemini", "OpenAI"])
    api_key = st.text_input(f"Enter {provider} API Key", type="password")
    
    st.divider()
    
    # Start New Chat Button
    if st.button("➕ Start New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # File Upload Section
    st.header("📁 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs (containing text, images, tables)", 
        type=["pdf"], 
        accept_multiple_files=True
    )

    if st.button("Process & Index PDFs", use_container_width=True):
        if not api_key:
            st.error(f"Please provide an API Key for {provider} first.")
        elif not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        else:
            with st.spinner("Extracting text, images, and tables..."):
                try:
                    status_msg = process_and_index_pdfs(uploaded_files, provider, api_key)
                    st.session_state.indexed = True
                    st.success(status_msg)
                except Exception as e:
                    st.error(f"Failed to process documents: {e}")

# 4. Render Conversation History
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# 5. User Input and Execution
user_query = st.chat_input("Ask anything about your uploaded PDFs...")

if user_query:
    if not api_key:
        st.error("Please enter your API key in the sidebar.")
        st.stop()

    if not st.session_state.indexed:
        st.warning("Please upload and index PDF documents before asking questions.")
        st.stop()

    # Display user input
    st.session_state.messages.append(HumanMessage(content=user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    # Initialize Agent & Stream Output
    with st.chat_message("assistant"):
        with st.spinner("Agent thinking & retrieving content..."):
            try:
                graph = create_multimodal_rag_graph(provider, api_key)
                
                # Execute LangGraph pipeline passing prior conversation state
                response = graph.invoke({"messages": st.session_state.messages})
                
                # Extract final message from agent
                final_response = response["messages"][-1].content
                st.markdown(final_response)
                
                # Save assistant response to session state
                st.session_state.messages.append(AIMessage(content=final_response))

            except Exception as e:
                st.error(f"An error occurred while processing your request: {e}")
