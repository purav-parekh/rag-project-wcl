import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# list of PDF files to be processed.
pdf_files = ["data/t20-pc.pdf","data/40-pc.pdf"]

# read pdfs
all_documents = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    documents = loader.load() # it reads the PDF, extracts the text, and returns a list of LangChain documents, ready for further processing.
    all_documents.extend(documents)
# loader = PyPDFLoader("data/t20-pc.pdf")
# document = loader.load() 

# create chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100) # chunk_size = max chars in each chunk, chunk_overlap = chars that overlap between chunks, helps maintain context btw neighboring chunks.
texts = text_splitter.split_documents(documents) # it splits the documents into smaller chunks for better processing.
print(f"Total chunks: {len(texts)}") # it prints the total number of chunks created from the documents.

# create embeddings
embeddings = OpenAIEmbeddings(openai_api_key = os.environ.get("OPEN_API_KEY")) # it creates embeddings for the text chunks using OpenAI's embedding model.
PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ.get("INDEX_NAME")) # it creates a Pinecone vector store from the text chunks and embeddings, allowing for efficient similarity search and retrieval of relevant information.