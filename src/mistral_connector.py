from logging import Logger
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, TypeAlias, TypeVar, overload
import httpx
from mistralai import UNSET, Mistral, OptionalNullable, RetryConfig
from mistralai import ChatCompletionResponse, AssistantMessage
from mistralai.utils.serializers import marshal_json
from mistralai.extra.exceptions import MistralClientException
from pydantic import BaseModel, ValidationError
from browser_use.llm.base import BaseChatModel
from browser_use.llm.exceptions import ModelProviderError
from browser_use.llm.messages import BaseMessage
from browser_use.llm.openai.serializer import OpenAIMessageSerializer
from browser_use.llm.schema import SchemaOptimizer
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage

T = TypeVar('T', bound=BaseModel)

MistralModel: TypeAlias = Literal[
    "mistral-medium-latest",
    "mistral-large-latest",
    "open-mistral-nemo",
    "mistral-small-latest",
    "magistral-medium-latest",
    "magistral-small-latest"
]

@dataclass
class ChatMistral(BaseChatModel):
    """
    A wrapper around the Mistral client (v1.0.0+) that implements the BaseChatModel protocol,
    enabling the browser-use Agent to be powered by Mistral AI models.
    """
    # --- Model Configuration ---
    model: MistralModel | str
    
    # --- Model Parameters ---
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: float | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    n: int | None = None
    prompt_mode: Literal["reasoning"] | None = None
    safe_prompt: bool | None = None

    # --- Client Initialization Parameters ---
    api_key: str | None = None
    server: str | None = None
    server_url: str | httpx.URL | None = None
    url_params: Dict[str, str] | None = None
    client: httpx.Client | None = None
    async_client: httpx.AsyncClient | None = None
    retry_config: RetryConfig | None = None
    timeout_ms: int | None = None
    debug_logger: Logger | None = None

    reasoning_models: list[str] | None = field(
		default_factory=lambda: [
			"magistral-medium-latest",
            "magistral-small-latest"
		]
	)

    @property
    def provider(self) -> str:
        return 'mistral'

    def _get_client_params(self) -> dict[str, Any]:
        """Prepare client parameters dictionary."""
		# Define base client params
        base_params = {
			'api_key': self.api_key if self.api_key else os.environ.get("MISTRAL_API_KEY"),
            'server': self.server,
			'server_url': self.server_url,
            'client': self.client,
            'async_client': self.async_client,
            'retry_config': self.retry_config,
            'timeout_ms': self.timeout_ms
		}
		# Create client_params dict with non-None values
        client_params = {k: v for k, v in base_params.items() if v is not None}
        
        return client_params
    
    def get_client(self) -> Mistral:
        """
		Returns a Mistral client.

		Returns:
			Mistral: An instance of the Mistral client.
		"""
        client_params = self._get_client_params()
        return Mistral(**client_params)

    @property
    def name(self) -> str:
        return str(self.model)

    def _get_model_params(self) -> dict[str, Any]:
            model_params: dict[str, Any] = {}
            
            if self.temperature is not None:
                model_params['temperature'] = self.temperature

            if self.top_p is not None:
                model_params['top_p'] = self.top_p

            if self.max_tokens is not None:
                model_params['max_tokens'] = self.max_tokens

            if self.seed is not None:
                model_params['seed'] = self.seed

            if self.presence_penalty is not None:
                model_params['presence_penalty'] = self.presence_penalty

            if self.frequency_penalty is not None:
                model_params['frequency_penalty'] = self.frequency_penalty

            if self.n is not None:
                model_params['n'] = self.n

            if self.safe_prompt is not None:
                model_params['safe_prompt'] = self.safe_prompt

            if self.reasoning_models and any(str(m).lower() in str(self.model).lower() for m in self.reasoning_models):
                if self.prompt_mode is not None:
                    model_params['prompt_mode'] = self.prompt_mode
            
            return model_params
    
    def _get_usage(self, response: ChatCompletionResponse) -> ChatInvokeUsage | None:
        """Extracts usage data from the Mistral API response."""
        if response.usage:
            return ChatInvokeUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                prompt_cached_tokens=None,
                prompt_cache_creation_tokens=None,
                prompt_image_tokens=None
            )
        return None

    @overload
    async def ainvoke(self, messages: list[BaseMessage], output_format: None = None) -> ChatInvokeCompletion[str]: ...

    @overload
    async def ainvoke(self, messages: list[BaseMessage], output_format: type[T]) -> ChatInvokeCompletion[T]: ...
    
    async def ainvoke(
        self, messages: list[BaseMessage], output_format: type[T] | None = None
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        """
		Invoke the model with the given messages.

		Args:
			messages: List of chat messages
			output_format: Optional Pydantic model class for structured output

		Returns:
			Either a string response or an instance of output_format
		"""

        mistral_messages: list[AssistantMessage] = OpenAIMessageSerializer.serialize_messages(messages)
        
        client = self.get_client()

        model_params: dict[str, Any] = self._get_model_params()

        try:
            if output_format is None:
                response = await client.chat.complete_async(
                    model=self.model, messages=mistral_messages, **model_params
                )
                usage=self._get_usage(response)
                return ChatInvokeCompletion(
                    completion=response.choices[0].message.content or '',
                    usage=usage
                )
            else:
                schema = SchemaOptimizer.create_optimized_json_schema(output_format)
                tools = [{
                    "type": "function",
                    "function": {
                        "name": output_format.__name__, # A consistent name for the agent's function
                        "description": f'Extract information in the format of {output_format.__name__}',
                        "parameters": schema,
                    },
                }]
                response = await client.chat.complete_async(
                    model=self.model,
                    messages=mistral_messages,
                    tools=tools,
                    tool_choice="required",
                    **model_params
                )

                tool_calls = response.choices[0].message.tool_calls
                if not tool_calls:
                    raise ModelProviderError(
                        message="Model failed to return a tool call for structured output.",
                        model=self.name,
                    )

                parsed_response = output_format.model_validate_json(tool_calls[0].function.arguments)
                usage = self._get_usage(response)

                return ChatInvokeCompletion(
                    completion=parsed_response,
                    usage=usage,
                )
        except MistralClientException as e:
            raise ModelProviderError(
                message=f"Mistral API Error: {str(e)}",
                status_code=getattr(e, 'response', None) and getattr(e.response, 'status_code', None),
                model=self.name,
            ) from e
        except Exception as e:
            raise ModelProviderError(message=str(e), model=self.name) from e