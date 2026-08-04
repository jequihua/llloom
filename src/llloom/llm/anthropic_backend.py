"""Anthropic Claude ``ModelBackend`` adapter.

Optional provider adapter for model-backed ``claim_extract`` ingestion.
Install via the extra:

    pip install "llloom[anthropic]"

The adapter implements the :class:`llloom.llm.ModelBackend` protocol.
It receives the deterministic prompt assembled by
:class:`llloom.llm.LLMInvoke` and calls ``client.messages.create(...)``
on the official Anthropic Python SDK. Output is plain text — strict
YAML per the contract in ``04_specification/component_contracts.md``
— and is parsed by the existing
:func:`llloom.llm.output.parse_claim_extraction_output` parser.

The adapter is single-turn: no streaming, no message batches, no
tools, no web search / files / hosted retrieval, no computer use,
no background mode, no multi-turn conversation state. The prompt is
the only context the model sees.

The adapter does not access workspace state, files, journals, pages,
commentary, spine prose, the search sidecar, or the graph sidecar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_SYSTEM_PROMPT = """\
You are a source-grounded claim extractor for the llloom package.

Emit YAML only. No Markdown fences. No commentary. No hidden reasoning.
Do not explain your output. Do not add surrounding text.

The top-level object must be a mapping with exactly one key, "claims",
whose value is a list of candidate claim entries. If you are not
confident about any claim, emit an empty list:

  claims: []

Never invent source text. Every "claim_text" and every "excerpt_hash"
must be grounded in the provided source. Use exact source excerpts
only.

Each claim entry must follow this shape:

  claims:
    - entity_id: <stable machine id>
      entity_type: <concept|person|organization|event|method>
      display_name: <human-readable name>
      claim_id: <stable claim id>
      claim_kind: <definition|assertion|finding|quantity>
      claim_text: <one sentence in the source's voice>
      locator:
        locator_type: markdown_prose_v1
        heading_path: [<heading>, <subheading>]
        paragraph_index: <1-based paragraph index under the heading>
        sentence_start: <1-based sentence start>
        sentence_end: <1-based sentence end>
      render_target:
        page_id: <page id>
        block_id: <block id within that page>
      excerpt_hash: <sha256 of the exact verbatim excerpt, if known>
      status: verified

"render_target" and "excerpt_hash" and "status" are optional.
"locator" is required. Emit only the locator_type that is supported by
the current ingest source class:

- for narrative sources (source_class with locator markdown_prose_v1 or
  legal_act_v1), emit only that narrative locator_type; never emit
  code_v1
- for an explicit code-backed claim_extract ingest (source_class with
  locator code_v1), emit code_v1 only for either:
  - declaration-level spans corresponding to real code entities such as
    class, function, method, type, interface, trait, enum, or struct
    definitions, or
  - attached explanation spans, which means either (a) the contiguous
    line-comment block immediately above one of those declarations
    (no blank line between the comment block and the declaration), or
    (b) a Python triple-quoted docstring on the line immediately below
    a class, function, or async-function declaration
- do not emit code_v1 for detached comments, free-floating comments,
  or arbitrary code-body spans

If you cannot meet this contract exactly, emit:

  claims: []
"""


_DEFAULT_MAX_OUTPUT_TOKENS = 4096


class AnthropicBackendError(Exception):
    """Raised for adapter-level failures that originate inside llloom
    rather than the Anthropic SDK.

    Typical causes: the ``anthropic`` optional dependency is not
    installed, or the SDK returned a response with no extractable
    text output.
    """


@dataclass
class AnthropicModelBackend:
    """Anthropic Claude backend conforming to :class:`ModelBackend`.

    The SDK is imported lazily on first ``generate`` call so that
    importing this module does not require the optional dependency.
    """

    model: str
    api_key: str | None = None
    timeout: float | None = None
    max_output_tokens: int = _DEFAULT_MAX_OUTPUT_TOKENS
    temperature: float | None = None
    system_prompt: str = _SYSTEM_PROMPT

    @property
    def identifier(self) -> str:
        return f"anthropic/{self.model}"

    def generate(self, prompt: str) -> str:
        client = self._build_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "system": self.system_prompt,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        message = client.messages.create(**kwargs)
        return _extract_output_text(message)

    def _build_client(self):
        try:
            from anthropic import Anthropic  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise AnthropicBackendError(
                "the Anthropic optional dependency is not installed; "
                "install it with: pip install \"llloom[anthropic]\""
            ) from exc
        client_kwargs: dict[str, Any] = {}
        if self.api_key is not None:
            client_kwargs["api_key"] = self.api_key
        if self.timeout is not None:
            client_kwargs["timeout"] = self.timeout
        return Anthropic(**client_kwargs)


def _extract_output_text(message: Any) -> str:
    """Return the concatenated text body from an Anthropic Messages
    API response.

    Walks ``message.content`` and collects ``text`` from every block
    whose ``type == "text"`` in the order the SDK returned them.
    Non-text blocks (tool use, images, etc.) are silently ignored.
    Raises :class:`AnthropicBackendError` if no text output can be
    recovered. Never silently returns the empty string — an empty
    response is an adapter-level failure, not a clean zero-candidate
    signal. (An empty YAML body from the model, by contrast, is
    parsed cleanly as zero candidates.)
    """
    content = getattr(message, "content", None)
    if not content:
        raise AnthropicBackendError(
            "Anthropic response had no content blocks; "
            "refusing to return an empty string to the harness"
        )
    collected: list[str] = []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type != "text":
            continue
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            collected.append(text)
    if not collected:
        raise AnthropicBackendError(
            "Anthropic response had no extractable text output; "
            "refusing to return an empty string to the harness"
        )
    return "".join(collected)
