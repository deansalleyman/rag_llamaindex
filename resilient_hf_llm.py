"""A streaming-safe HuggingFace Inference API LLM.

Why this exists
---------------
The upstream ``HuggingFaceInferenceAPI`` creates one ``AsyncInferenceClient`` in
``__init__`` and then calls ``await self._async_client.close()`` at the end of
every ``astream_chat`` / ``astream_complete`` (see
``llama_index/llms/huggingface_api/base.py``). Because the client is shared, the
*second* streaming call on the same instance raises::

    RuntimeError: Cannot send a request, as the client has been closed.

A ReAct multi-agent workflow makes many streaming calls on a single shared LLM
(one per reasoning step, across every agent), so it breaks right after the first
stream. This subclass refreshes the async client before each streaming call, so
every stream starts with an open client.

Note: this assumes streaming calls on a given instance are sequential (true for
a single-query ReAct handoff flow). It is not safe for concurrent streams that
share one instance.
"""

from typing import Any, Sequence

from huggingface_hub import AsyncInferenceClient
from llama_index.core.base.llms.types import ChatResponseAsyncGen, CompletionResponseAsyncGen
from llama_index.core.llms import ChatMessage
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI


class ResilientHuggingFaceInferenceAPI(HuggingFaceInferenceAPI):
    """HuggingFaceInferenceAPI that survives multiple streaming calls."""

    def _refresh_async_client(self) -> None:
        """Replace the async client so a previously closed one is never reused."""
        self._async_client = AsyncInferenceClient(**self._get_inference_client_kwargs())

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        self._refresh_async_client()
        return await super().astream_chat(messages, **kwargs)

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        self._refresh_async_client()
        return await super().astream_complete(prompt, formatted=formatted, **kwargs)
