from dataclasses import dataclass

from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from lagom.environment import Env

from autogen_chat.protocols.i_azure_openai_service import IAzureOpenAIService


class AzureOpenAIServiceEnv(Env):
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_api_version: str
    azure_openai_deployed_model_name: str


@dataclass
class AzureOpenAIService(IAzureOpenAIService):
    env: AzureOpenAIServiceEnv

    def get_model(self) -> AzureOpenAIChatCompletionClient:
        return AzureOpenAIChatCompletionClient(
            azure_endpoint=self.env.azure_openai_endpoint,
            api_key=self.env.azure_openai_api_key,
            model=self.env.azure_openai_deployed_model_name,
            api_version=self.env.azure_openai_api_version,
        )
