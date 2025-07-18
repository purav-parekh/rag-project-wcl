import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

load_dotenv()

openai_api_key = os.environ.get("OPENAI_API_KEY")

# chat_history = []

embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore(
    index_name=os.environ.get("INDEX_NAME"),
    embedding=embeddings
)

chat = ChatOpenAI(model="gpt-4.1-nano", temperature=0, verbose=True)
qa = ConversationalRetrievalChain.from_llm(
    llm=chat,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
)

st.title("💬 WCL Conversational Chatbot")
st.caption("🚀 Have a conversation about WCL! Your chat history will be remembered during this session.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for user, bot in st.session_state.chat_history:
    st.chat_message("user").write(user)
    st.chat_message("assistant").write(bot)

# user_input = st.text_input("Your question:",key="input")

# if user_input:
#     with st.spinner("Thinking..."):
#         try:
#             res = qa({"question": user_input, "chat_history": st.session_state.chat_history})
#             answer = res["answer"]
#             st.session_state.chat_history.append((user_input, answer))
#             st.markdown(f"**You:** {user_input}")
#             st.markdown(f"**Bot:** {answer}")
#             # st.write("Answer:", res['result'])
#         except Exception as e:
#             st.error(f"An error occurred: {e}")

with st.sidebar:
    # openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Go to WCL Website](https://www.wclinc.com/)"
    # "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    # "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"


if prompt := st.chat_input("Ask a question about WCL!"):
    with st.spinner("Thinking..."):
        try:
            msg = qa({"question": prompt, "chat_history": st.session_state.chat_history})
            answer = msg["answer"]
            st.session_state.chat_history.append((prompt, answer))
            st.chat_message("assistant").write(answer)
        except Exception as e:
            st.error(f"An error occurred: {e}")
