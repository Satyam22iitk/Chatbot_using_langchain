import os
import openai
import streamlit as st
from datetime import datetime
from streamlit.logger import get_logger
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

logger = get_logger('Langchain-Chatbot')

# Add DeepSeek configuration at the top
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODELS = ["deepseek-chat", "deepseek-coder", "deepseek-llm"]

def get_llm_client(api_key, provider="openai"):
    """Get LLM client based on provider"""
    if provider.lower() == "deepseek":
        return openai.OpenAI(
            api_key=api_key,
            base_url=DEEPSEEK_BASE_URL
        )
    else:
        return openai.OpenAI(api_key=api_key)

#decorator
def enable_chat_history(func):
    # Check for any API key (OpenAI or DeepSeek)
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY"):

        # to clear chat history after swtching chatbot
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

def choose_custom_api_key():
    """Choose between OpenAI and DeepSeek API keys"""
    api_provider = st.sidebar.radio(
        label="API Provider",
        options=["OpenAI", "DeepSeek"],
        key="SELECTED_API_PROVIDER"
    )
    
    if api_provider == "OpenAI":
        api_key = st.sidebar.text_input(
            label="OpenAI API Key",
            type="password",
            placeholder="sk-...",
            key="SELECTED_OPENAI_API_KEY"
        )
        if not api_key:
            st.error("Please add your OpenAI API key to continue.")
            st.info("Obtain your key from: https://platform.openai.com/account/api-keys")
            st.stop()
        return "openai", api_key, None
        
    else:  # DeepSeek
        api_key = st.sidebar.text_input(
            label="DeepSeek API Key",
            type="password",
            placeholder="sk-...",
            key="SELECTED_DEEPSEEK_API_KEY"
        )
        if not api_key:
            st.error("Please add your DeepSeek API key to continue.")
            st.info("Obtain your key from: https://platform.deepseek.com/")
            st.stop()
        
        # For DeepSeek, we can directly return the model since it's compatible
        model = st.sidebar.selectbox(
            label="DeepSeek Model",
            options=DEEPSEEK_MODELS,
            key="SELECTED_DEEPSEEK_MODEL"
        )
        return "deepseek", api_key, model

def configure_llm():
    available_llms = ["gpt-4.1-mini", "llama3.2:3b", "use your api key", "deepseek-chat"]
    llm_opt = st.sidebar.radio(
        label="LLM",
        options=available_llms,
        key="SELECTED_LLM"
    )

    if llm_opt == "llama3.2:3b":
        llm = ChatOllama(model="llama3.2", base_url=st.secrets["OLLAMA_ENDPOINT"])
    elif llm_opt == "gpt-4.1-mini":
        llm = ChatOpenAI(model_name=llm_opt, temperature=0, streaming=True, 
                        api_key=st.secrets.get("OPENAI_API_KEY"))
    elif llm_opt == "deepseek-chat":
        # DeepSeek integration - uses same ChatOpenAI class with different parameters
        llm = ChatOpenAI(
            model_name="deepseek-chat",
            temperature=0,
            streaming=True,
            api_key=st.secrets.get("DEEPSEEK_API_KEY"),
            base_url=DEEPSEEK_BASE_URL
        )
    else:
        provider, api_key, model = choose_custom_api_key()
        if provider == "deepseek":
            llm = ChatOpenAI(
                model_name=model,
                temperature=0,
                streaming=True,
                api_key=api_key,
                base_url=DEEPSEEK_BASE_URL
            )
        else:
            llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0, 
                           streaming=True, api_key=api_key)
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
