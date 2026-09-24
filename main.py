from crewai import Crew, Process
from agents import support_agent, compliance_agent
from tasks import create_tasks

def run_banking_assistant(customer_question: str):
    # Build the tasks for this specific question
    support_task, compliance_task = create_tasks(customer_question)

    # Assemble the crew
    crew = Crew(
        agents=[support_agent, compliance_agent],
        tasks=[support_task, compliance_task],
        process=Process.sequential,   # run tasks in order: support -> compliance
        verbose=True
    )

    # Run it
    result = crew.kickoff()
    return result


if __name__ == "__main__":
    question = "What documents do I need to apply for a personal loan, and is there a processing fee?"

    print("=" * 60)
    print(f"CUSTOMER QUESTION: {question}")
    print("=" * 60)

    final_answer = run_banking_assistant(question)

    print("\n" + "=" * 60)
    print("FINAL COMPLIANCE-APPROVED ANSWER:")
    print("=" * 60)
    print(final_answer)