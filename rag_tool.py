import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

load_dotenv()

# -----------------------------
# Reconnect to the existing ChromaDB (built in Step 3)
# -----------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# -----------------------------
# Define the input schema for the tool
# -----------------------------
class BankingKnowledgeInput(BaseModel):
    query: str = Field(..., description="The banking question to search for in the knowledge base")


# -----------------------------
# Define the custom CrewAI Tool
# -----------------------------
class BankingKnowledgeTool(BaseTool):
    name: str = "Banking Knowledge Base Search"
    description: str = (
        "Searches the bank's internal policy documents (savings accounts, "
        "loans, and fraud/security policies) to answer customer questions. "
        "Always use this tool before answering any question about bank "
        "policies, fees, eligibility, or procedures."
    )
    args_schema: type[BaseModel] = BankingKnowledgeInput

    def _run(self, query: str) -> str:
        results = vectorstore.similarity_search(query, k=3)
        if not results:
            return "No relevant information found in the knowledge base."

        combined = "\n\n---\n\n".join([doc.page_content for doc in results])
        return combined


# -----------------------------
# Quick standalone test
# -----------------------------
if __name__ == "__main__":
    tool = BankingKnowledgeTool()
    test_result = tool._run("What documents are needed for a personal loan?")
    print("Tool Test Result:\n")
    print(test_result)