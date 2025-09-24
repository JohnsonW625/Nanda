#!/usr/bin/env python3
"""
A CrewAI twin agent wrapped with Nanda Adapter.

This version uses Anthropic Claude (via langchain_anthropic.ChatAnthropic)
and exposes the agent to Nanda for discoverability and A2A communication.
"""

import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import FileWriterTool, FileReadTool
from langchain_anthropic import ChatAnthropic
from nanda_adapter import NANDA


def create_johnson_agent():
    """Return a function that Nanda will use to process messages."""

    # Load Anthropic LLM (Claude)
    llm = ChatAnthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-3-haiku-20240307"
    )

    # Define the twin agent
    twin_agent = Agent(
        role="Johnson's Twin Agent",
        goal="Answer questions about Johnson Wang’s background and expertise.",
        backstory=(
            "You are Johnson Wang, a master’s student at Harvard "
            "who excels in academics and is passionate about AI and biology. "
            "You answer questions concisely, in the first person, using knowledge "
            "from the provided background file."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[FileReadTool(file_path='./information.txt')]
    )

    # Define the writer agent
    writer_agent = Agent(
        role="Formal Response Writer",
        goal="Take Johnson’s draft and polish it into a formal, professional reply.",
        backstory=(
            "You are a professional communicator who improves clarity and style, "
            "keeping content factual and concise."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[FileWriterTool()]
    )

    # This inner function will be called for each incoming Nanda message
    def johnson_response(message_text: str) -> str:
        # Create tasks dynamically
        research_task = Task(
            description=f"""Question: {message_text}

            Your task (on behalf of Johnson Wang):
            1. Read `information.txt` to learn Johnson's background.
            2. Draft a first-person answer (2-3 sentences).
            3. Add a short supporting paragraph (3-4 sentences).
            """,
            expected_output="A draft answer + supporting explanation",
            agent=twin_agent
        )

        writing_task = Task(
            description=(
                f"Polish this draft into a final reply:\n\n{message_text}\n\n"
                "Your task:\n"
                "1. Convert it into a concise, formal, first-person reply (2–4 sentences).\n"
                "2. Keep it factual. Do not invent new information.\n"
                "3. Save final reply to 'johnson_reply.md'."
            ),
            expected_output="Final polished reply",
            agent=writer_agent
        )

        crew = Crew(
            agents=[twin_agent, writer_agent],
            tasks=[research_task, writing_task],
            process=Process.sequential,
            verbose=True
        )

        result = crew.kickoff()
        return str(result)

    return johnson_response


if __name__ == "__main__":
    # Wrap the CrewAI function with Nanda
    nanda = NANDA(create_johnson_agent())

    # Start the Nanda server
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    domain = os.getenv("DOMAIN_NAME", "localhost")

    nanda.start_server_api(anthropic_key, domain)
