from crewai import Task
from agents import support_agent, compliance_agent

def create_tasks(customer_question: str):
    """
    Builds the two tasks for a given customer question.
    Takes the question as a parameter so we can reuse this
    for any question, not just one hardcoded example.
    """

    # -----------------------------
    # Task 1: Draft an answer using RAG
    # -----------------------------
    support_task = Task(
        description=(
            f"A customer has asked: '{customer_question}'\n\n"
            "Use the Banking Knowledge Base Search tool to find the exact "
            "relevant policy information. Do not answer from memory or "
            "general knowledge. Base your answer strictly on what the tool "
            "returns. If the tool does not return relevant information, "
            "clearly state that you don't have that information rather "
            "than guessing."
        ),
        expected_output=(
            "A clear, customer-friendly draft answer to the question, "
            "grounded strictly in the retrieved policy information, "
            "including specific numbers, fees, or eligibility criteria "
            "where relevant."
        ),
        agent=support_agent
    )

    # -----------------------------
    # Task 2: Compliance review of the draft
    # -----------------------------
    compliance_task = Task(
        description=(
            "Review the draft answer provided by the support agent. "
            "Verify every claim against the Banking Knowledge Base Search "
            "tool yourself — do not simply trust the draft. Correct any "
            "inaccuracies, remove any overpromising language, and add a "
            "short compliance disclaimer if the answer involves rates, "
            "fees, or approval decisions (e.g., 'Rates and terms are "
            "subject to change and final approval is subject to bank "
            "review.')."
        ),
        expected_output=(
            "The final, compliance-approved answer ready to send to the "
            "customer, written in a clear and professional tone."
        ),
        agent=compliance_agent,
        context=[support_task]   # gives this task access to support_task's output
    )

    return support_task, compliance_task