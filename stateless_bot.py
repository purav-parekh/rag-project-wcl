import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# establishes a connection to existing Pinecone index where the vector representations (embeddings) is stored.
embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore(index_name=os.environ.get("INDEX_NAME"), embedding=embeddings)

# langchain workflows
chat = ChatOpenAI(model="gpt-4.1-nano", temperature=0,verbose=True)
qa = RetrievalQA.from_chain_type(
    llm=chat,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
)

# res = qa.invoke("What is the time limit for each over bowled?")
# print(res)

res = qa.invoke("What are the differences in the playing conditions for 20 overs vs 40 overs? Can you only elaborate on the differences?")
print(res)