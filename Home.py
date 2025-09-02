import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title="AI Chatbots with LangChain",
    page_icon="🤖",
    layout="wide"
)

# --- Title Section ---
st.markdown(
    """
    <style>
        .title {
            font-size:40px !important;
            font-weight:700 !important;
            color:#4CAF50;
        }
        .subtitle {
            font-size:20px !important;
            color:#555;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="title">🤖 AI Chatbots with LangChain</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">College Project | Built using Streamlit & LangChain 🚀</p>', unsafe_allow_html=True)

st.write("---")

# --- About Section ---
st.info(
    """
    **LangChain** is a framework that makes it easier to build powerful applications using 
    large language models (LLMs).  
    This project demonstrates how different types of **chatbots** can be built with LangChain.
    """
)

# --- Chatbot Examples ---
st.subheader("📂 Available Chatbot Implementations")

st.markdown("""
- 💬 **Basic Chatbot** → Simple interactive conversations with an LLM.  
- 🧠 **Context Aware Chatbot** → Remembers previous chats and responds with context.  
- 📄 **Chat with Your Documents** → Upload PDFs or notes and ask questions.    
""")

st.success("👉 Use the sidebar to navigate and try different chatbots!")
