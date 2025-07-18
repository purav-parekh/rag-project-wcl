import os
import warnings
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

chat_history = []
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful expert that understans and knows all the playing conditions of cricket and more importantly the Washington Cricket League. You are able to answer questions based on the provided context."),
    ("human", "Context: {context}\nChat history: {chat_history}\nQuestion: {question}\n\nAnswer the question based on the context provided. If you don't know the answer, say 'I don't know'. If the question is not related to cricket, say 'This question is not related to cricket'.")
])

if __name__ == "__main__":
    # establishes a connection to existing Pinecone index where the vector representations (embeddings) is stored.
    embeddings = OpenAIEmbeddings(openai_api_key = os.environ.get("OPEN_API_KEY"))
    vectorstore = PineconeVectorStore(index_name=os.environ.get("INDEX_NAME"), embedding=embeddings)

    # langchain workflows
    chat = ChatOpenAI(model="gpt-4.1-nano", temperature=0, verbose=True)
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        combine_docs_chain_kwargs={"prompt": prompt_template},
        return_source_documents=False,
    )

    res = qa({"question": "What are the playing conditions for changing the bowling ends in T20 and in T40?", "chat_history": chat_history})
    print(res)

    history = (res["question"], res["answer"])
    chat_history.append(history)

    print("..................................................................")

    # res = qa({"question": "Can you please elaborate more on it?", "chat_history": chat_history})
    # print(res)