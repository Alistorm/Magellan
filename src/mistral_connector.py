import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar, overload
from mistralai import Mistral
from mistralai import ChatCompletionResponse, AssistantMessage
from mistralai.extra.exceptions import MistralClientException
from pydantic import BaseModel, ValidationError
from browser_use.llm.base import BaseChatModel
from browser_use.llm.exceptions import ModelProviderError
from browser_use.llm.messages import BaseMessage
from browser_use.llm.openai.serializer import OpenAIMessageSerializer
from browser_use.llm.schema import SchemaOptimizer
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage

T = TypeVar('T', bound=BaseModel)

@dataclass
class ChatMistral(BaseChatModel):
    """
    A wrapper around the Mistral client (v1.0.0+) that implements the BaseChatModel protocol,
    enabling the browser-use Agent to be powered by Mistral AI models.
    """
    # --- Model Configuration ---
    model: str = "mistral-large-latest"
    
    # --- Model Parameters ---
    temperature: float = 0.0
    max_tokens: int | None = 4096
    top_p: float = 1.0
    random_seed: int | None = None
    safe_prompt: bool = False

    # --- Client Initialization Parameters ---
    api_key: str | None = None
    endpoint: str = "https://api.mistral.ai"
    timeout: int = 120*1000
    max_retries: int = 5

    @property
    def provider(self) -> str:
        return 'mistral'

    def get_client(self) -> Mistral:
        """Initializes and returns the unified Mistral client."""
        api_key = self.api_key or os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError(
                "Mistral API key not found. Provide it via the 'api_key' parameter "
                "or set the MISTRAL_API_KEY environment variable."
            )
        return Mistral(
            api_key=api_key,
            timeout_ms=self.timeout,
        )

    @property
    def name(self) -> str:
        return str(self.model)

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

    def _pydantic_to_mistral_tool(self, model: type[BaseModel]) -> dict[str, Any]:
        """Converts a Pydantic model into a Mistral tool schema."""
        schema = SchemaOptimizer.create_optimized_json_schema(model)
        return {
            "type": "function",
            "function": {
                "name": "agent_output",
                "description": "The structured output for the agent's action.",
                "parameters": schema,
            },
        }

    @overload
    async def ainvoke(self, messages: list[BaseMessage], output_format: None = None) -> ChatInvokeCompletion[str]: ...

    @overload
    async def ainvoke(self, messages: list[BaseMessage], output_format: type[T]) -> ChatInvokeCompletion[T]: ...

    # --- UPDATED ainvoke METHOD ---
    async def ainvoke(
        self, messages: list[BaseMessage], output_format: type[T] | None = None
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        """Invoke the Mistral model with the given messages using the v1.0.0+ API."""
        mistral_messages: list[AssistantMessage] = OpenAIMessageSerializer.serialize_messages(messages)
        client = self.get_client()

        model_params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "random_seed": self.random_seed,
            "safe_prompt": self.safe_prompt,
        }
        model_params = {k: v for k, v in model_params.items() if v is not None}

        try:
            if output_format is None:
                # --- Path 1: Simple Text Response ---
                response = await client.chat.complete_async(
                    model=self.model, messages=mistral_messages, **model_params
                )
                completion_text = response.choices[0].message.content or ""
                return ChatInvokeCompletion(
                    completion=completion_text, usage=self._get_usage(response)
                )
            else:
                # --- Path 2: Structured Output via Tool Calling ---
                tool = self._pydantic_to_mistral_tool(output_format)
                response = await client.chat.complete_async(
                    model=self.model,
                    messages=mistral_messages,
                    tools=[tool],
                    tool_choice="required",
                    **model_params,
                )

                tool_calls = response.choices[0].message.tool_calls
                if not tool_calls:
                    raise ModelProviderError(
                        message="Model failed to return a tool call for structured output.", model=self.name
                    )

                json_string = tool_calls[0].function.arguments
                
                try:
                    parsed = output_format.model_validate_json(json_string)
                except ValidationError as e:
                     raise ModelProviderError(
                        message=f"Failed to validate model output against schema: {e}", model=self.name
                    ) from e

                return ChatInvokeCompletion(completion=parsed, usage=self._get_usage(response))
        except MistralClientException as e:
            raise ModelProviderError(
                message=f"Mistral API Error: {str(e)}",
                status_code=getattr(e, 'response', None) and getattr(e.response, 'status_code', None),
                model=self.name,
            ) from e
        except Exception as e:
            raise ModelProviderError(message=str(e), model=self.name) from e