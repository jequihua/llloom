"""OpenAI GPT `ModelBackend` adapter.

Optional provider adapter for model-backed ``claim_extract`` ingestion.
Install via the extra:

    pip install "llloom[openai]"

The adapter implements the :class:`llloom.llm.ModelBackend` protocol.
It receives the deterministic prompt assembled by
:class:`llloom.llm.LLMInvoke` and calls
``client.responses.create(...)`` on the official OpenAI Python SDK.
Output is plain text — strict YAML per the contract in
``04_specification/component_contracts.md`` — and is parsed by the
existing :func:`llloom.llm.output.parse_claim_extraction_output`
parser. The adapter does not perform tool calls, web search, file
search, code interpreter, background mode, function calling, or
hosted retrieval; the prompt is the only context the model sees.

The adapter does not access workspace state, files, journals, pages,
commentary, spine prose, the search sidecar, or the graph sidecar.
It has access only to the prompt it is handed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_INSTRUCTIONS = """\
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


_DEFAULT_MODEL = "gpt-5.4"


class OpenAIBackendError(Exception):
    """Raised for adapter-level failures that originate inside llloom
    rather than the OpenAI SDK.

    Typical causes: the ``openai`` optional dependency is not
    installed, or the SDK returned a response with no extractable
    text output.
    """


@dataclass
class OpenAIModelBackend:
    """OpenAI GPT backend conforming to :class:`ModelBackend`.

    The SDK is imported lazily on first ``generate`` call so that
    importing this module does not require the optional dependency.
    """

    model: str = _DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    timeout: float | None = None
    reasoning_effort: str | None = None
    max_output_tokens: int | None = None
    instructions: str = _INSTRUCTIONS

    @property
    def identifier(self) -> str:
        return f"openai/{self.model}"

    def generate(self, prompt: str) -> str:
        client = self._build_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": self.instructions,
            "input": prompt,
        }
        if self.max_output_tokens is not None:
            kwargs["max_output_tokens"] = self.max_output_tokens
        if self.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        response = client.responses.create(**kwargs)
        return _extract_output_text(response)

    def _build_client(self):
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise OpenAIBackendError(
                "the OpenAI optional dependency is not installed; "
                "install it with: pip install \"llloom[openai]\""
            ) from exc
        client_kwargs: dict[str, Any] = {}
        if self.api_key is not None:
            client_kwargs["api_key"] = self.api_key
        if self.base_url is not None:
            client_kwargs["base_url"] = self.base_url
        if self.timeout is not None:
            client_kwargs["timeout"] = self.timeout
        return OpenAI(**client_kwargs)


def _extract_output_text(response: Any) -> str:
    """Return the text body from an OpenAI Responses API response.

    Prefers ``response.output_text`` (the SDK's canonical accessor).
    Falls back to a minimal structural walk only if ``output_text`` is
    absent; raises :class:`OpenAIBackendError` if no text can be
    recovered. Never silently returns the empty string — an empty
    response is an adapter-level failure, not a clean zero-candidate
    signal. (An empty YAML body from the model, by contrast, is
    parsed cleanly as zero candidates.)
    """
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text:
        return text
    output = getattr(response, "output", None)
    if output:
        collected: list[str] = []
        for item in output:
            content = getattr(item, "content", None) or []
            for part in content:
                piece = getattr(part, "text", None)
                if isinstance(piece, str) and piece:
                    collected.append(piece)
        if collected:
            return "".join(collected)
    raise OpenAIBackendError(
        "OpenAI response had no extractable text output; "
        "refusing to return an empty string to the harness"
    )
