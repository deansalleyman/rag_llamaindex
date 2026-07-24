"""Offline test for the streaming-safe HuggingFace inference LLM.

Regression test for a library bug: the upstream ``HuggingFaceInferenceAPI``
closes its shared ``_async_client`` at the end of every ``astream_chat`` /
``astream_complete`` (llama_index/llms/huggingface_api/base.py), but the client
is created once in ``__init__``. The second streaming call on the same instance
then raises "Cannot send a request, as the client has been closed." A ReAct
multi-agent workflow makes many streaming calls on one shared LLM, so it breaks
after the first stream.

The fix must refresh the async client before each streaming call. This test
patches the HF client classes so no network call happens.
"""

import asyncio
from unittest.mock import MagicMock, patch

from llama_index.core.llms import ChatMessage


def test_astream_chat_refreshes_closed_async_client():
    """astream_chat must swap in a fresh async client (so a closed one is replaced)."""
    with patch("llama_index.llms.huggingface_api.base.InferenceClient"), patch(
        "llama_index.llms.huggingface_api.base.AsyncInferenceClient"
    ), patch(
        "resilient_hf_llm.AsyncInferenceClient",
        side_effect=lambda **kwargs: MagicMock(name="fresh_async_client"),
    ):
        from resilient_hf_llm import ResilientHuggingFaceInferenceAPI

        llm = ResilientHuggingFaceInferenceAPI(model_name="test/model", token="x")
        stale_client = llm._async_client  # stands in for a client closed by a prior stream

        # Returning the generator must have already refreshed the client.
        asyncio.run(llm.astream_chat([ChatMessage(role="user", content="hi")]))

        assert llm._async_client is not stale_client
