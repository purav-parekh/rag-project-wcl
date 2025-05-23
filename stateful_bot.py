import os
import warnings
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

chat_history = []

if __name__ == "__main__":
    # establishes a connection to existing Pinecone index where the vector representations (embeddings) is stored.
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(index_name=os.environ.get("INDEX_NAME"), embedding=embeddings)

    # langchain workflows
    chat = ChatOpenAI(model="gpt-4.1-nano", temperature=0, verbose=True)
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        # return_source_documents=True,
    )

    res = qa({"question": "What are the playing conditions for a player arriving late?", "chat_history": chat_history})
    print(res)

    history = (res["question"], res["answer"])
    chat_history.append(history)

    res = qa({"question": "Can you please elaborate more on it?", "chat_history": chat_history})
    print(res)