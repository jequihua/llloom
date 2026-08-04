"""Unit tests for the OpenAI GPT ``ModelBackend`` adapter.

The adapter is an optional provider behind the ``llloom[openai]``
extra. These tests never perform network I/O: they inject a fake
``OpenAI`` client via monkeypatching and they never require the real
SDK to be installed.
"""

from __future__ import annotations

import sys
import types

import pytest

from llloom.llm.openai_backend import (
    OpenAIBackendError,
    OpenAIModelBackend,
    _extract_output_text,
)


class _FakeResponses:
    def __init__(self, output_text: str = "", output_shape: list | None = None) -> None:
        self._text = output_text
        self._output = output_shape
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        obj = types.SimpleNamespace()
        if self._text:
            obj.output_text = self._text
        if self._output is not None:
            obj.output = self._output
        return obj


class _FakeClient:
    def __init__(self, responses: _FakeResponses) -> None:
        self.responses = responses


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch, client: _FakeClient) -> None:
    """Put a fake ``openai`` module in ``sys.modules`` so the lazy
    import inside ``OpenAIModelBackend._build_client`` returns it."""
    fake = types.ModuleType("openai")

    def _factory(**kwargs):
        fake.last_kwargs = kwargs  # type: ignore[attr-defined]
        return client

    fake.OpenAI = _factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake)


def test_import_does_not_require_openai_sdk() -> None:
    """Importing the adapter module must not import ``openai``.

    The real SDK may or may not be installed on the developer's
    machine; this test covers both environments uniformly.
    """
    # Already imported above at module top; re-importing is cheap and
    # asserts the invariant.
    import importlib

    mod = importlib.import_module("llloom.llm.openai_backend")
    assert hasattr(mod, "OpenAIModelBackend")
    # The adapter must not have eagerly bound the SDK.
    assert "openai" not in mod.__dict__ or not isinstance(
        mod.__dict__.get("openai"), types.ModuleType
    )


def test_identifier_includes_model() -> None:
    backend = OpenAIModelBackend(model="gpt-test-1")
    assert backend.identifier == "openai/gpt-test-1"


def test_default_model_is_gpt_5_4() -> None:
    """The adapter default must track the current OpenAI guidance.

    Callers who want `gpt-5.4-mini` or `gpt-5.4-nano` pass the model
    explicitly; this test guards the default against silent drift.
    """
    backend = OpenAIModelBackend()
    assert backend.model == "gpt-5.4"
    assert backend.identifier == "openai/gpt-5.4"


def test_explicit_model_override_preserved() -> None:
    for override in ("gpt-5.4-mini", "gpt-5.4-nano", "gpt-test-1"):
        backend = OpenAIModelBackend(model=override)
        assert backend.model == override
        assert backend.identifier == f"openai/{override}"


def test_generate_calls_responses_create_with_expected_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = _FakeResponses(output_text="claims: []\n")
    _install_fake_openai(monkeypatch, _FakeClient(responses))

    backend = OpenAIModelBackend(
        model="gpt-test-1",
        api_key="sk-test",
        max_output_tokens=1024,
    )
    out = backend.generate("# prompt body")
    assert out == "claims: []\n"

    assert len(responses.calls) == 1
    call = responses.calls[0]
    assert call["model"] == "gpt-test-1"
    assert call["input"] == "# prompt body"
    assert "instructions" in call
    assert call["max_output_tokens"] == 1024
    # No tools, no web/file search, no function calling, no background mode.
    for forbidden in ("tools", "tool_choice", "web_search", "file_search",
                      "code_interpreter", "background", "functions"):
        assert forbidden not in call, f"{forbidden!r} leaked into responses.create"


def test_instructions_require_strict_yaml_and_no_fences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = _FakeResponses(output_text="claims: []\n")
    _install_fake_openai(monkeypatch, _FakeClient(responses))
    backend = OpenAIModelBackend(model="gpt-test-1")
    backend.generate("prompt")
    instructions = responses.calls[0]["instructions"]
    # Contract smells the prompt must carry.
    assert "YAML only" in instructions
    assert "No Markdown fences" in instructions
    assert "Never invent source text" in instructions
    assert "claims: []" in instructions


def test_generate_returns_output_text_when_present() -> None:
    response = types.SimpleNamespace(output_text="claims: []\n")
    assert _extract_output_text(response) == "claims: []\n"


def test_generate_falls_back_to_output_content_when_output_text_empty() -> None:
    response = types.SimpleNamespace(
        output=[
            types.SimpleNamespace(
                content=[types.SimpleNamespace(text="claims:\n  - x: 1\n")]
            )
        ],
    )
    assert _extract_output_text(response) == "claims:\n  - x: 1\n"


def test_generate_raises_when_no_text_output() -> None:
    response = types.SimpleNamespace(output_text="", output=[])
    with pytest.raises(OpenAIBackendError):
        _extract_output_text(response)


def test_generate_raises_backend_error_when_sdk_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate a machine without ``openai`` installed.

    We stash any real ``openai`` module and replace it with a fake
    whose ``OpenAI`` attribute raises ``ImportError``; the adapter
    must convert that into a package-owned ``OpenAIBackendError``
    that names the install extra.
    """
    # Make ``from openai import OpenAI`` fail.
    fake = types.ModuleType("openai")

    def _raise(**kwargs):
        raise ImportError("no openai")

    fake.__getattr__ = lambda name: (_ for _ in ()).throw(  # type: ignore[attr-defined]
        ImportError("no openai SDK")
    )
    monkeypatch.setitem(sys.modules, "openai", fake)

    backend = OpenAIModelBackend(model="gpt-test-1")
    with pytest.raises(OpenAIBackendError) as excinfo:
        backend.generate("prompt")
    assert "llloom[openai]" in str(excinfo.value)


def test_client_receives_only_configured_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = _FakeResponses(output_text="claims: []\n")
    fake = types.ModuleType("openai")
    captured: dict = {}

    def _factory(**kwargs):
        captured.update(kwargs)
        return _FakeClient(responses)

    fake.OpenAI = _factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake)

    backend = OpenAIModelBackend(
        model="gpt-test-1",
        api_key="sk-test",
        base_url="https://api.example.com/v1",
        timeout=5.0,
    )
    backend.generate("prompt")
    assert captured["api_key"] == "sk-test"
    assert captured["base_url"] == "https://api.example.com/v1"
    assert captured["timeout"] == 5.0


def test_adapter_does_not_access_workspace_state(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The adapter receives only the prompt string from the harness.

    It must not read files, journals, or any workspace path. The
    fake client records everything it is called with; the only
    string supplied to ``input=`` is the prompt the caller passed.
    """
    responses = _FakeResponses(output_text="claims: []\n")
    _install_fake_openai(monkeypatch, _FakeClient(responses))
    backend = OpenAIModelBackend(model="gpt-test-1")
    prompt = "# operation: ingest\n\n## source src.test [markdown_prose] hash=sha256:abc\nbody"
    backend.generate(prompt)
    call = responses.calls[0]
    assert call["input"] == prompt
    # The adapter has no workspace handle. If it ever grew one this
    # test would need to change; that is the point.
    assert not hasattr(backend, "workspace")
    assert not hasattr(backend, "read_file")
