import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load the OpenAI API key from .env
load_dotenv()

# -----------------------------
# 1. Load all documents from knowledge_base folder
# -----------------------------
loader = DirectoryLoader(
    "knowledge_base",
    glob="*.txt",
    loader_cls=TextLoader
)
documents = loader.load()
print(f"Loaded {len(documents)} documents.")

# -----------------------------
# 2. Split documents into smaller chunks
# -----------------------------
# Why: LLMs work better with smaller, focused chunks of text
# rather than one huge document, so retrieval is more precise.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # ~500 characters per chunk
    chunk_overlap=50     # slight overlap so context isn't cut off mid-sentence
)
chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks.")

# -----------------------------
# 3. Create embeddings and store in ChromaDB
# -----------------------------
# Embeddings turn text into number vectors so we can search by meaning,
# not just keyword matching.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"   # saves the database to disk
)

print("Vector database created and saved to 'chroma_db' folder.")

# -----------------------------
# 4. Quick test: search the knowledge base
# -----------------------------
query = "What is the minimum balance for a savings account?"
results = vectorstore.similarity_search(query, k=2)

print("\n--- Test Query ---")
print(f"Query: {query}\n")
for i, doc in enumerate(results, 1):
    print(f"Result {i}:")
    print(doc.page_content)
    print("-" * 40)
