import os
import openai
import streamlit as st
from datetime import datetime
from streamlit.logger import get_logger
from langchain_openai import ChatOpenAI
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

logger = get_logger('Langchain-Chatbot')

# DeepSeek Configuration
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODELS = ["deepseek-chat", "deepseek-coder", "deepseek-llm"]

#decorator
def enable_chat_history(func):
    if os.environ.get("DEEPSEEK_API_KEY"):

        # to clear chat history after switching chatbot
        current_page = func.__qualname__
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = current_page
        if st.session_state["current_page"] != current_page:
            try:
                st.cache_resource.clear()
                del st.session_state["current_page"]
                del st.session_state["messages"]
            except:
                pass

        # to show chat history on ui
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

    def execute(*args, **kwargs):
        func(*args, **kwargs)
    return execute

def display_msg(msg, author):
    """Method to display message on the UI

    Args:
        msg (str): message to display
        author (str): author of the message -user/assistant
    """
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)

def choose_deepseek_key():
    """Get DeepSeek API key from user"""
    deepseek_api_key = st.sidebar.text_input(
        label="DeepSeek API Key",
        type="password",
        placeholder="sk-...",
        key="DEEPSEEK_API_KEY_INPUT"
    )
    if not deepseek_api_key:
        st.error("Please add your DeepSeek API key to continue.")
        st.info("Obtain your key from: https://platform.deepseek.com/")
        st.stop()
    
    # Test the API key
    try:
        client = openai.OpenAI(api_key=deepseek_api_key, base_url=DEEPSEEK_BASE_URL)
        client.models.list()  # Simple test call
    except Exception as e:
        st.error(f"Invalid DeepSeek API key: {str(e)}")
        st.stop()
    
    return deepseek_api_key

def configure_llm():
    """Configure DeepSeek LLM only"""
    available_models = DEEPSEEK_MODELS
    
    # Use secrets if available, otherwise get from user input
    deepseek_api_key = st.secrets.get("DEEPSEEK_API_KEY")
    
    if not deepseek_api_key:
        deepseek_api_key = choose_deepseek_key()
    
    # Model selection
    selected_model = st.sidebar.selectbox(
        label="DeepSeek Model",
        options=available_models,
        key="SELECTED_DEEPSEEK_MODEL",
        index=0  # Default to deepseek-chat
    )
    
    # Configure DeepSeek LLM using ChatOpenAI with custom base_url
    llm = ChatOpenAI(
        model_name=selected_model,
        temperature=0,
        streaming=True,
        api_key=deepseek_api_key,
        base_url=DEEPSEEK_BASE_URL,
        max_retries=3
    )
    
    return llm

def print_qa(cls, question, answer):
    log_str = "\nUsecase: {}\nQuestion: {}\nAnswer: {}\n" + "------"*10
    logger.info(log_str.format(cls.__name__, question, answer))

@st.cache_resource
def configure_embedding_model():
    embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return embedding_model

def sync_st_session():
    for k, v in st.session_state.items():
        st.session_state[k] = v

# Initialize OpenAI client for direct API calls (if needed)
def get_deepseek_client():
    """Get DeepSeek client for direct API calls"""
    api_key = st.secrets.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    return openai.OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
