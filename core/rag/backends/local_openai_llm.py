from typing import Any, Optional

from openai import OpenAI
from pydantic import Field, PrivateAttr

from llama_index.core.base.llms.types import CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.llms.custom import CustomLLM


class LocalOpenAICompatibleLLM(CustomLLM):
    """Minimal OpenAI-compatible LlamaIndex LLM for local LM Studio-style endpoints."""

    model_name: str = Field(description="The local model identifier.")
    base_url: str = Field(description="OpenAI-compatible base URL.")
    api_key: str = Field(default="lm-studio", description="API key for the local endpoint.")
    temperature: float = Field(default=0.0, description="Sampling temperature.")
    context_window: int = Field(default=16384, description="Approximate context window.")
    num_output: int = Field(default=768, description="Maximum response tokens.")
    system_prompt: Optional[str] = Field(default=None, description="Optional system instruction.")

    _client: OpenAI = PrivateAttr()

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    @classmethod
    def class_name(cls) -> str:
        return "local_openai_compatible_llm"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            is_chat_model=False,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=float(kwargs.get("temperature", self.temperature)),
            max_tokens=int(kwargs.get("max_tokens", self.num_output)),
        )
        text = ""
        if response.choices:
            text = str(getattr(response.choices[0].message, "content", "") or "")
        raw = response.model_dump() if hasattr(response, "model_dump") else response
        return CompletionResponse(
            text=text.strip(),
            raw=raw,
            additional_kwargs={"model_name": self.model_name},
        )

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        response = self.complete(prompt, formatted=formatted, **kwargs)

        def gen() -> CompletionResponseGen:
            yield CompletionResponse(
                text=response.text,
                delta=response.text,
                raw=response.raw,
                additional_kwargs=response.additional_kwargs,
            )

        return gen()
