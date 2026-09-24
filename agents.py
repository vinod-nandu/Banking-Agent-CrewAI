import os
from dotenv import load_dotenv
from crewai import Agent, LLM
from rag_tool import BankingKnowledgeTool

load_dotenv()

# -----------------------------
# The LLM that powers both agents
# -----------------------------
llm = LLM(
    model="gpt-4o-mini",   # cheap, fast, good enough for this use case
    temperature=0.2         # low temperature = more factual, less "creative"
)

# -----------------------------
# The RAG tool both agents can use
# -----------------------------
banking_tool = BankingKnowledgeTool()

# -----------------------------
# Agent 1: Customer Support Agent
# -----------------------------
support_agent = Agent(
    role="Bank Customer Support Specialist",
    goal=(
        "Answer customer questions about bank policies, accounts, loans, "
        "and security accurately by searching the bank's official knowledge base."
    ),
    backstory=(
        "You are an experienced customer support representative at a retail bank. "
        "You always base your answers strictly on official bank policy documents, "
        "never on assumptions or general knowledge, since incorrect banking "
        "information can cause real financial harm to customers."
    ),
    tools=[banking_tool],
    llm=llm,
    verbose=True
)

# -----------------------------
# Agent 2: Compliance Review Agent
# -----------------------------
compliance_agent = Agent(
    role="Banking Compliance Reviewer",
    goal=(
        "Review customer support answers for accuracy, ensure they are fully "
        "grounded in the bank's official policy, and add necessary disclaimers."
    ),
    backstory=(
        "You are a compliance officer at a retail bank. Your job is to catch "
        "any overpromising, vague, or unsupported claims in customer-facing "
        "answers before they go out, and to ensure regulatory disclaimers "
        "are included where relevant (e.g., rates are subject to change, "
        "approval is not guaranteed)."
    ),
    tools=[banking_tool],
    llm=llm,
    verbose=True
)

print("Agents created successfully.")