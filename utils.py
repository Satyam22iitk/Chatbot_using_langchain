import os
import openai
import streamlit as st
import time
from datetime import datetime
from streamlit.logger import get_logger
from langchain_openai import ChatOpenAI
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from openai import APIStatusError, APIConnectionError, RateLimitError

logger = get_logger('Langchain-Chatbot')

# DeepSeek Configuration
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODELS = ["deepseek-chat", "deepseek-coder", "deepseek-llm"]

def enable_chat_history(func):
    if os.environ.get("DEEPSEEK_API_KEY") or st.secrets.get("DEEPSEEK_API_KEY"):
        # ... [rest of your existing enable_chat_history code] ...
    def execute(*args, **kwargs):
        func(*args, **kwargs)
    return execute

def display_msg(msg, author):
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)

def choose_deepseek_key():
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
    return deepseek_api_key

def configure_llm():
    """Configure DeepSeek LLM with retry mechanism"""
    available_models = DEEPSEEK_MODELS
    
    # Get API key from secrets or user input
    deepseek_api_key = st.secrets.get("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        deepseek_api_key = choose_deepseek_key()
    
    # Model selection
    selected_model = st.sidebar.selectbox(
        label="DeepSeek Model",
        options=available_models,
        key="SELECTED_DEEPSEEK_MODEL",
        index=0
    )
    
    # Configure DeepSeek LLM with retry settings
    llm = ChatOpenAI(
        model_name=selected_model,
        temperature=0,
        streaming=True,
        api_key=deepseek_api_key,
        base_url=DEEPSEEK_BASE_URL,
        max_retries=3,  # Add retry mechanism
        request_timeout=30  # Increase timeout
    )
    
    return llm

def execute_with_retry(func, max_retries=3, *args, **kwargs):
    """Wrapper function with retry logic for API calls"""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except (APIStatusError, APIConnectionError, RateLimitError) as e:
            if attempt == max_retries - 1:
                raise e
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"API error (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            raise e

# ... [rest of your existing functions] ...
