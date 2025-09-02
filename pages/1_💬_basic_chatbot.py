import streamlit as st
from utils import enable_chat_history, display_msg, configure_llm, execute_with_retry
from streaming import StreamHandler
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

class BasicChatbot:

    def main(self):
        st.header('Basic Chatbot')
        enable_chat_history(self.main)

        # Initialize LLM
        llm = configure_llm()
        
        # Setup memory and chain
        memory = ConversationBufferMemory()
        prompt = PromptTemplate.from_template("""
        You are a helpful AI assistant. Have a conversation with the human.

        Current conversation:
        {history}
        Human: {input}
        AI:""")
        
        chain = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=prompt,
            verbose=True
        )

        if prompt := st.chat_input(placeholder='Ask me anything!'):
            display_msg(prompt, 'user')
            
            with st.chat_message('assistant'):
                st_cb = StreamHandler(st.empty())
                
                try:
                    # Use retry wrapper for the API call
                    result = execute_with_retry(
                        chain.invoke,
                        max_retries=3,
                        input={"input": prompt},
                        config={'callbacks': [st_cb]}
                    )
                    
                    # Display final response
                    response = result['response']
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except APIStatusError as e:
                    st.error("DeepSeek API is currently unavailable. Please try again later.")
                    st.info("This could be due to server maintenance or high traffic.")
                except APIConnectionError as e:
                    st.error("Connection to DeepSeek API failed. Please check your internet connection.")
                except RateLimitError as e:
                    st.error("Rate limit exceeded. Please wait a moment before trying again.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")

obj = BasicChatbot()
obj.main()
