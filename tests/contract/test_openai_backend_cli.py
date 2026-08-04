"""Contract tests for the CLI integration of the OpenAI GPT backend.

These tests prove the CLI plumbs provider flags correctly without
touching the network or the real ``openai`` SDK.
"""

from __future__ import annotations

import io
import sys
import types
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from llloom.claims.models import Locator
from llloom.cli import main as cli_main
from llloom.llm.harness import LLMInvoke
from llloom.llm.openai_backend import OpenAIModelBackend
from llloom.ops.ingest import SeedClaim
from llloom.state.journal import OperationJournal
from llloom.workspace.layout import Workspace


SOURCE = """\
# Article

## Methods

Complementarity prioritizes sites that add features not already represented in the selected set.
"""


def _seed_workspace(tmp_path: Path) -> tuple[Workspace, Path]:
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "article.md"
    src.write_text(SOURCE, encoding="utf-8")
    return ws, src


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch, client_obj) -> None:
    fake = types.ModuleType("openai")
    fake.OpenAI = lambda **kwargs: client_obj  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake)


def _install_missing_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a machine without the openai SDK by ensuring
    ``import openai`` raises ``ImportError``."""

    class _Raiser:
        def find_spec(self, name, path=None, target=None):
            if name == "openai":
                raise ImportError("no openai SDK")
            return None

    monkeypatch.delitem(sys.modules, "openai", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Raiser(), *sys.meta_path])


# --- 1. CLI with --model-provider openai builds the adapter -------------


def test_cli_ingest_with_openai_provider_builds_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws, src = _seed_workspace(tmp_path)

    captured: dict = {}

    import llloom.cli as cli_module

    real_ingest = cli_module.ingest

    def _spy_ingest(workspace, path, **kwargs):
        captured["llm"] = kwargs.get("llm")
        return real_ingest(workspace, path, **kwargs)

    monkeypatch.setattr(cli_module, "ingest", _spy_ingest)

    # Provide a fake openai module so _build_harness's early probe succeeds.
    fake = types.ModuleType("openai")
    fake.OpenAI = lambda **kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake)

    # The adapter would call the SDK on generate, but this test never
    # exercises generate: the default ingest with NullModel runs
    # through because markdown_prose's default policy would normally
    # invoke the harness. We need the harness to return empty output.
    # Route generate through a fake that yields an empty yaml body.
    captured_calls: list = []

    def _fake_generate(self, prompt: str) -> str:  # noqa: ARG001
        captured_calls.append(prompt)
        return "claims: []\n"

    monkeypatch.setattr(OpenAIModelBackend, "generate", _fake_generate)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli_main([
            "--root", str(ws.root),
            "ingest", str(src),
            "--model-provider", "openai",
            "--model", "gpt-test-1",
        ])

    assert rc == 0, buf.getvalue()
    harness = captured["llm"]
    assert isinstance(harness, LLMInvoke)
    # The harness wraps an OpenAIModelBackend.
    backend = harness._model  # type: ignore[attr-defined]
    assert isinstance(backend, OpenAIModelBackend)
    assert backend.model == "gpt-test-1"
    # The backend was actually invoked during ingest.
    assert len(captured_calls) == 1


# --- 2. Default CLI ingest does not build the OpenAI adapter ------------


def test_cli_default_ingest_does_not_construct_openai_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws, src = _seed_workspace(tmp_path)

    import llloom.cli as cli_module

    captured: dict = {}
    real_ingest = cli_module.ingest

    def _spy_ingest(workspace, path, **kwargs):
        captured["llm"] = kwargs.get("llm")
        return real_ingest(workspace, path, **kwargs)

    monkeypatch.setattr(cli_module, "ingest", _spy_ingest)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli_main(["--root", str(ws.root), "ingest", str(src)])
    assert rc == 0, buf.getvalue()
    # Default CLI path passes llm=None so ingest uses its internal NullModel.
    assert captured["llm"] is None


# --- 3. Missing optional dependency produces a clear CLI error ----------


def test_cli_openai_missing_sdk_refuses_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws, src = _seed_workspace(tmp_path)
    _install_missing_openai(monkeypatch)

    err_buf = io.StringIO()
    out_buf = io.StringIO()
    with redirect_stderr(err_buf), redirect_stdout(out_buf):
        rc = cli_main([
            "--root", str(ws.root),
            "ingest", str(src),
            "--model-provider", "openai",
            "--model", "gpt-test-1",
        ])
    assert rc == 2
    assert "llloom[openai]" in err_buf.getvalue()
    # No JSON ingest result was printed.
    assert out_buf.getvalue() == ""


# --- 4. index_only ingest cuts off BEFORE any backend call --------------


def _wire_index_only(ws: Workspace) -> None:
    (ws.schema / "source_classes.yaml").write_text(
        "classes:\n"
        "  markdown_prose:\n"
        "    locator: markdown_prose_v1\n"
        "  sensitive:\n"
        "    locator: markdown_prose_v1\n",
        encoding="utf-8",
    )
    (ws.schema / "ingest_policies.yaml").write_text(
        "policies:\n"
        "  markdown_prose: claim_extract_and_view_render\n"
        "  sensitive: index_only\n"
        "defaults:\n  unknown: deny\n",
        encoding="utf-8",
    )


def test_index_only_ingest_with_openai_provider_does_not_invoke_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = Workspace.init(tmp_path)
    _wire_index_only(ws)
    src = ws.raw_sources / "contract.md"
    src.write_text(
        "# Contract\n\nNet-30 with a 2% early-payment discount.\n",
        encoding="utf-8",
    )

    # Install a fake openai module so the provider probe succeeds.
    fake = types.ModuleType("openai")
    fake.OpenAI = lambda **kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake)

    # Any call to generate would be a safety violation: index_only must
    # return BEFORE any harness invocation happens.
    def _fail_generate(self, prompt: str) -> str:  # noqa: ARG001
        raise AssertionError(
            "OpenAIModelBackend.generate must not be called for index_only ingest"
        )

    monkeypatch.setattr(OpenAIModelBackend, "generate", _fail_generate)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli_main([
            "--root", str(ws.root),
            "ingest", str(src),
            "--source-class", "sensitive",
            "--model-provider", "openai",
            "--model", "gpt-test-1",
        ])
    assert rc == 0, buf.getvalue()


# --- 5. Invocation-log persistence stays at one entry per model ingest --


class _FakeResponses:
    def __init__(self, text: str) -> None:
        self._text = text

    def create(self, **kwargs):  # noqa: ARG002
        return types.SimpleNamespace(output_text=self._text)


class _FakeClient:
    def __init__(self, text: str) -> None:
        self.responses = _FakeResponses(text)


def test_invocation_log_persists_once_per_openai_ingest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws, src = _seed_workspace(tmp_path)

    # Fake openai module. The fake client returns a YAML body with
    # one valid claim candidate grounded in the seeded source.
    yaml_body = (
        "claims:\n"
        "  - entity_id: concept.complementarity\n"
        "    entity_type: concept\n"
        "    display_name: Complementarity\n"
        "    claim_id: c.cmp.1\n"
        "    claim_kind: definition\n"
        "    claim_text: Complementarity prioritizes sites that add features not already represented in the selected set.\n"
        "    locator:\n"
        "      locator_type: markdown_prose_v1\n"
        "      heading_path: [Methods]\n"
        "      paragraph_index: 1\n"
        "      sentence_start: 1\n"
        "      sentence_end: 1\n"
    )
    _install_fake_openai(monkeypatch, _FakeClient(yaml_body))

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli_main([
            "--root", str(ws.root),
            "ingest", str(src),
            "--model-provider", "openai",
            "--model", "gpt-test-1",
        ])
    assert rc == 0, buf.getvalue()

    journal = OperationJournal(ws)
    entries = [e for e in journal.iter_entries() if e.op_kind == "ingest"]
    assert len(entries) == 1
    entry = entries[0]
    logs = entry.invocation_logs
    assert len(logs) == 1, f"expected exactly one invocation log, got {logs}"
    log = logs[0]
    # Summary only — no raw source text.
    assert log["model_identifier"] == "openai/gpt-test-1"
    for read in log.get("read_inputs", []):
        for key, value in read.items():
            if isinstance(value, str):
                assert "Complementarity prioritizes" not in value, (
                    f"raw source text leaked into invocation log {log}"
                )
