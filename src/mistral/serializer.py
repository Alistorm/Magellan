from typing import overload

import mistralai
from browser_use.llm.messages import (
    AssistantMessage,
    BaseMessage,
    ContentPartImageParam,
    ContentPartRefusalParam,
    ContentPartTextParam,
    SystemMessage,
    ToolCall,
    UserMessage,
)
from mistralai import Function, ImageURL, ImageURLChunk, TextChunk


class MistralMessageSerializer:
    """Serializer for converting between custom message types and Mistral message param types."""

    @staticmethod
    def _serialize_content_part_text(part: ContentPartTextParam) -> TextChunk:
        return TextChunk(text=part.text, type="text")

    @staticmethod
    def _serialize_content_part_image(part: ContentPartImageParam) -> ImageURLChunk:
        return ImageURLChunk(
            image_url=ImageURL(url=part.image_url.url, detail=part.image_url.detail),
            type="image_url",
        )

    @staticmethod
    def _serialize_content_part_refusal(part: ContentPartRefusalParam) -> dict:
        return {"refusal": part.refusal, "type": "refusal"}

    @staticmethod
    def _serialize_user_content(
        content: str | list[ContentPartTextParam | ContentPartImageParam],
    ) -> str | list[TextChunk | ImageURLChunk]:
        """Serialize content for user messages (text and images allowed)."""
        if isinstance(content, str):
            return content

        serialized_parts: list[TextChunk | ImageURLChunk] = []
        for part in content:
            if part.type == "text":
                serialized_parts.append(MistralMessageSerializer._serialize_content_part_text(part))
            elif part.type == "image_url":
                serialized_parts.append(MistralMessageSerializer._serialize_content_part_image(part))
        return serialized_parts

    @staticmethod
    def _serialize_system_content(
        content: str | list[ContentPartTextParam],
    ) -> str | list[TextChunk]:
        """Serialize content for system messages (text only)."""
        if isinstance(content, str):
            return content

        serialized_parts: list[TextChunk] = []
        for part in content:
            if part.type == "text":
                serialized_parts.append(MistralMessageSerializer._serialize_content_part_text(part))
        return serialized_parts

    @staticmethod
    def _serialize_assistant_content(
        content: str | list[ContentPartTextParam | ContentPartRefusalParam] | None,
    ) -> str | list[TextChunk | dict] | None:
        """Serialize content for assistant messages (text and refusal allowed)."""
        if content is None:
            return None
        if isinstance(content, str):
            return content

        serialized_parts: list[TextChunk | dict] = []
        for part in content:
            if part.type == "text":
                serialized_parts.append(MistralMessageSerializer._serialize_content_part_text(part))
            elif part.type == "refusal":
                serialized_parts.append(MistralMessageSerializer._serialize_content_part_refusal(part))
        return serialized_parts

    @staticmethod
    def _serialize_tool_call(tool_call: ToolCall) -> mistralai.ToolCall:
        return mistralai.ToolCall(
            id=tool_call.id,
            function=Function(name=tool_call.function.name, arguments=tool_call.function.arguments),
            type="function",
        )

    # endregion

    # region - Serialize overloads
    @overload
    @staticmethod
    def serialize(message: UserMessage) -> mistralai.UserMessage: ...

    @overload
    @staticmethod
    def serialize(message: SystemMessage) -> mistralai.SystemMessage: ...

    @overload
    @staticmethod
    def serialize(message: AssistantMessage) -> mistralai.AssistantMessage: ...

    @staticmethod
    def serialize(message: BaseMessage) -> mistralai.Messages:
        """Serialize a custom message to an OpenAI message param."""

        if isinstance(message, UserMessage):
            user_result: mistralai.UserMessage = {
                "role": "user",
                "content": MistralMessageSerializer._serialize_user_content(message.content),
            }
            if message.name is not None:
                user_result["name"] = message.name
            return user_result

        elif isinstance(message, SystemMessage):
            system_result: mistralai.SystemMessage = {
                "role": "system",
                "content": MistralMessageSerializer._serialize_system_content(message.content),
            }
            if message.name is not None:
                system_result["name"] = message.name
            return system_result

        elif isinstance(message, AssistantMessage):
            # Handle content serialization
            content = None
            if message.content is not None:
                content = MistralMessageSerializer._serialize_assistant_content(message.content)

            assistant_result: mistralai.AssistantMessage = {"role": "assistant"}

            # Only add content if it's not None
            if content is not None:
                assistant_result["content"] = content

            if message.name is not None:
                assistant_result["name"] = message.name
            if message.refusal is not None:
                assistant_result["refusal"] = message.refusal
            if message.tool_calls:
                assistant_result["tool_calls"] = [
                    MistralMessageSerializer._serialize_tool_call(tc) for tc in message.tool_calls
                ]

            return assistant_result

        else:
            raise ValueError(f"Unknown message type: {type(message)}")

    @staticmethod
    def serialize_messages(messages: list[BaseMessage]) -> list[mistralai.Messages]:
        return [MistralMessageSerializer.serialize(m) for m in messages]
