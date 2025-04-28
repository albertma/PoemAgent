import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily
import os

async def main():

    # Create an OpenAI model client.
    model_client = OpenAIChatCompletionClient(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"), # Optional if you have an OPENAI_API_KEY env variable set.
        base_url="https://api.deepseek.com/v1",
        temperature=0.7,
        model_info={
                    "vision": False,
                    "function_calling": False,
                    "json_output": False,
                    "family": ModelFamily.R1,
                    "structured_output": True,
                },
        )

    # Create the primary agent.
    creating_agent = AssistantAgent(
        "creating",
        model_client=model_client,
        system_message="You are a helpful AI assistant.",
    )

    # Create the critic agent.
    critic_agent = AssistantAgent(
        "critic",
        model_client=model_client,
        system_message="Provide constructive feedback. Respond with from 1 to 100 marks to when your feedbacks are addressed.",
    )

    # Define a termination condition that stops the task if the critic approves.
    text_termination = TextMentionTermination("marks are 100")


    team = RoundRobinGroupChat(
        [creating_agent, critic_agent],
        termination_condition=text_termination,  # Use the bitwise OR operator to combine conditions.
    )

    await Console(team.run_stream(task="写一首关于暮春的短诗"))
    
    
if __name__ == "__main__":
    asyncio.run(main())

