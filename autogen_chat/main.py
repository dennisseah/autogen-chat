import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console

from autogen_chat.hosting import container
from autogen_chat.protocols.i_azure_openai_service import IAzureOpenAIService
from autogen_chat.tools.tools import (
    get_bank_account_id,
    get_investment_account_balance,
    get_saving_account_balance,
)

llm_client = container[IAzureOpenAIService].get_model()

customer_agent = AssistantAgent(
    "customer_agent",
    model_client=llm_client,
    description="A bank assistant.",
    tools=[get_bank_account_id],
    system_message="You are a helpful bank assistant who can assist customer "
    "with request. You need to get the customer bank account ID first and then "
    "ask the other agents for assistance. "
    "When customer's request is completed, you can respond with TERMINATE.",
)

investment_agent = AssistantAgent(
    "investment_agent",
    model_client=llm_client,
    description="An investment account agent.",
    tools=[get_investment_account_balance],
    system_message="You are an investment account agent who can provide "
    "information about the investment account balance.",
)

saving_account_agent = AssistantAgent(
    "saving_account_agent",
    model_client=llm_client,
    description="A saving account agent.",
    tools=[get_saving_account_balance],
    system_message="You are a saving account agent who can provide information "
    "about the saving account balance.",
)

termination = TextMentionTermination("TERMINATE")
group_chat = RoundRobinGroupChat(
    [customer_agent, investment_agent, saving_account_agent],
    termination_condition=termination,
)


async def main():
    await Console(
        group_chat.run_stream(
            task="Get my investment and saving account balance. And sum them up."
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
