"""Unit tests for the Anthropic Claude ``ModelBackend`` adapter.

The adapter is an optional provider behind the ``llloom[anthropic]``
extra. These tests never perform network I/O: they inject a fake
``Anthropic`` client via monkeypatching and they never require the
real SDK to be installed.
"""

from __future__ import annotations

import sys
import types

import pytest

from llloom.llm.anthropic_backend import (
    AnthropicBackendError,
    AnthropicModelBackend,
    _extract_output_text,
)


class _FakeMessages:
    def __init__(self, content: list | None = None) -> None:
        self._content = content if content is not None else []
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(content=self._content)


class _FakeClient:
    def __init__(self, messages: _FakeMessages) -> None:
        self.messages = messages


def _install_fake_anthropic(
    monkeypatch: pytest.MonkeyPatch, client: _FakeClient
) -> dict:
    """Put a fake ``anthropic`` module in ``sys.modules`` so the lazy
    import inside ``AnthropicModelBackend._build_client`` returns it.
    Returns a dict the test can inspect for the constructor kwargs."""
    fake = types.ModuleType("anthropic")
    captured: dict = {}

    def _factory(**kwargs):
        captured.update(kwargs)
        return client

    fake.Anthropic = _factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    return captured


def _text_block(text: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(type="text", text=text)


# --- 1. importing the adapter does not require the SDK ------------------


def test_import_does_not_require_anthropic_sdk() -> None:
    """Importing the adapter module must not import ``anthropic``.

    The real SDK may or may not be installed on the developer's
    machine; this test covers both environments uniformly.
    """
    import importlib

    mod = importlib.import_module("llloom.llm.anthropic_backend")
    assert hasattr(mod, "AnthropicModelBackend")
    # The adapter must not have eagerly bound the SDK.
    assert "anthropic" not in mod.__dict__ or not isinstance(
        mod.__dict__.get("anthropic"), types.ModuleType
    )


# --- 2. identifier exposes the configured model ------------------------


def test_identifier_includes_model() -> None:
    backend = AnthropicModelBackend(model="claude-test-1")
    assert backend.identifier == "anthropic/claude-test-1"


# --- 3. missing SDK raises a package-owned error -----------------------


def test_generate_raises_backend_error_when_sdk_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate a machine without ``anthropic`` installed.

    We replace the module with a fake whose ``Anthropic`` attribute
    raises ``ImportError``; the adapter must convert that into a
    package-owned ``AnthropicBackendError`` that names the install
    extra.
    """
    fake = types.ModuleType("anthropic")

    def _raise(**kwargs):  # noqa: ARG001
        raise ImportError("no anthropic")

    fake.__getattr__ = lambda name: (_ for _ in ()).throw(  # type: ignore[attr-defined]
        ImportError("no anthropic SDK")
    )
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    backend = AnthropicModelBackend(model="claude-test-1")
    with pytest.raises(AnthropicBackendError) as excinfo:
        backend.generate("prompt")
    assert "llloom[anthropic]" in str(excinfo.value)


# --- 4. messages.create receives the expected kwargs -------------------


def test_generate_calls_messages_create_with_expected_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = _FakeMessages(content=[_text_block("claims: []\n")])
    _install_fake_anthropic(monkeypatch, _FakeClient(messages))

    backend = AnthropicModelBackend(
        model="claude-test-1",
        api_key="sk-test",
        max_output_tokens=2048,
    )
    out = backend.generate("# prompt body")
    assert out == "claims: []\n"

    assert len(messages.calls) == 1
    call = messages.calls[0]
    assert call["model"] == "claude-test-1"
    assert call["max_tokens"] == 2048
    assert isinstance(call["system"], str)
    assert call["messages"] == [{"role": "user", "content": "# prompt body"}]
    # No tools, no streaming, no batches, no web search, no computer
    # use, no background, no caching directives.
    for forbidden in (
        "tools",
        "tool_choice",
        "stream",
        "stop_sequences",
        "metadata",
        "service_tier",
    ):
        # ``stop_sequences`` and ``metadata`` are valid SDK kwargs in
        # general but this adapter intentionally never passes them.
        assert forbidden not in call, (
            f"{forbidden!r} leaked into messages.create"
        )


# --- 5. system prompt requires strict YAML / no fences / no commentary -


def test_system_prompt_requires_strict_yaml_and_no_fences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = _FakeMessages(content=[_text_block("claims: []\n")])
    _install_fake_anthropic(monkeypatch, _FakeClient(messages))

    backend = AnthropicModelBackend(model="claude-test-1")
    backend.generate("prompt")
    system = messages.calls[0]["system"]
    assert "YAML only" in system
    assert "No Markdown fences" in system
    assert "No commentary" in system
    assert "Never invent source text" in system
    assert "claims: []" in system
    # And the system prompt still forbids `code_v1` on narrative
    # sources (and admits it for explicit code-backed claim_extract
    # ingest only), matching the OpenAI adapter's contract.
    assert "narrative sources" in system
    assert "never emit\n  code_v1" in system or "never emit code_v1" in system


# --- 6. text extraction succeeds for a normal single text block --------


def test_extract_output_text_single_text_block() -> None:
    message = types.SimpleNamespace(content=[_text_block("claims: []\n")])
    assert _extract_output_text(message) == "claims: []\n"


# --- 7. text extraction concatenates multiple text blocks in order -----


def test_extract_output_text_concatenates_text_blocks_in_order() -> None:
    message = types.SimpleNamespace(
        content=[
            _text_block("claims:\n"),
            _text_block("  - x: 1\n"),
            _text_block("  - y: 2\n"),
        ]
    )
    assert _extract_output_text(message) == "claims:\n  - x: 1\n  - y: 2\n"


# --- 8. non-text blocks are ignored ------------------------------------


def test_extract_output_text_ignores_non_text_blocks() -> None:
    """Tool-use / image / thinking blocks must be silently skipped;
    only ``text`` blocks contribute to the recovered output."""
    message = types.SimpleNamespace(
        content=[
            types.SimpleNamespace(type="tool_use", input={"foo": "bar"}),
            _text_block("claims: []\n"),
            types.SimpleNamespace(type="image", source="..."),
            types.SimpleNamespace(type="thinking", thinking="ignored reasoning"),
        ]
    )
    assert _extract_output_text(message) == "claims: []\n"


# --- 9. missing text output raises a package-owned error ---------------


def test_extract_output_text_raises_when_no_text() -> None:
    # Empty content list — adapter-level failure.
    with pytest.raises(AnthropicBackendError):
        _extract_output_text(types.SimpleNamespace(content=[]))
    # Content with only non-text blocks — adapter-level failure.
    with pytest.raises(AnthropicBackendError):
        _extract_output_text(
            types.SimpleNamespace(
                content=[types.SimpleNamespace(type="tool_use", input={})]
            )
        )
    # Content with a text-type block whose text is empty — adapter-level failure.
    with pytest.raises(AnthropicBackendError):
        _extract_output_text(
            types.SimpleNamespace(content=[_text_block("")])
        )


# --- 10. the adapter has no workspace surface beyond the prompt --------


def test_adapter_does_not_access_workspace_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adapter receives only the prompt string from the harness.

    It must not read files, journals, or any workspace path. The
    fake client records everything it is called with; the only
    string supplied to the user message is the prompt the caller
    passed.
    """
    messages = _FakeMessages(content=[_text_block("claims: []\n")])
    _install_fake_anthropic(monkeypatch, _FakeClient(messages))
    backend = AnthropicModelBackend(model="claude-test-1")
    prompt = (
        "# operation: ingest\n\n"
        "## source src.test [markdown_prose] hash=sha256:abc\n"
        "body"
    )
    backend.generate(prompt)
    call = messages.calls[0]
    assert call["messages"][0]["content"] == prompt
    # The adapter has no workspace handle. If it ever grew one this
    # test would need to change; that is the point.
    assert not hasattr(backend, "workspace")
    assert not hasattr(backend, "read_file")
    assert not hasattr(backend, "load_source")
