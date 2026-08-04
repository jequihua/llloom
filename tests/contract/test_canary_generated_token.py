"""Contract tests for the generated per-run canary token.

The fixed fixture token
(:data:`llloom.ops.lint.FIXED_CANARY_TOKEN`) already catches
deterministic regression leaks. The generated token adds release-
validation coverage by making leaks harder to accidentally satisfy
through hard-coding around the fixed token.

Contract:

- `generate_canary_token()` returns high-entropy strings prefixed
  `LLLOOM_CANARY_RUN_`.
- `lint(ws, generated_canary=True)` passes on a clean workspace.
- If the generated token appears in a rendered claim-block region,
  lint must flag it in `canary_hits` and `LintResult.passed` must
  be False.
- Same for persisted YAML surfaces (entity files, merge proposals,
  operation journals).
- The CLI plumbs `--generated-canary` through.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

import importlib

# Resolve the submodule explicitly: ``llloom.ops.__init__`` re-exports
# ``lint`` as a function, which shadows the submodule's attribute on
# the parent package. ``import_module`` returns the module object
# regardless of the parent-attribute binding.
lint_module = importlib.import_module("llloom.ops.lint")

from llloom.claims.models import Locator
from llloom.cli import main as cli_main
from llloom.ops.ingest import SeedClaim, ingest
from llloom.ops.lint import (
    FIXED_CANARY_TOKEN,
    GENERATED_CANARY_PREFIX,
    generate_canary_token,
    lint,
)
from llloom.state.journal import OperationJournal
from llloom.workspace.layout import Workspace


SOURCE = """\
# Article

## Methods

A canonical sentence in paragraph one.
"""

PAGE_TEMPLATE = """\
---
page_id: concept/example
page_class: concept
write_policy: mixed
status: rendered
---

<!-- llloom:claim-block id=claim_block.concept.example -->
## Example

Original rendered content with no canary.
<!-- /llloom:claim-block -->

<!-- llloom:commentary id=commentary.concept.example owner=human -->

Human commentary.

<!-- /llloom:commentary -->
"""


@pytest.fixture
def deterministic_token(monkeypatch: pytest.MonkeyPatch) -> str:
    """Return the same generated token every time lint asks for one.

    The real `generate_canary_token` uses `secrets.token_hex`, which
    is non-deterministic. Tests patch it so the canary they plant
    matches the canary lint generates.
    """
    token = f"{GENERATED_CANARY_PREFIX}" + ("b" * 32)
    monkeypatch.setattr(lint_module, "generate_canary_token", lambda: token)
    return token


def _seed_clean_workspace(tmp_path: Path) -> Workspace:
    return Workspace.init(tmp_path)


def _seed_claim_extract_workspace(tmp_path: Path) -> tuple[Workspace, Path]:
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "article.md"
    src.write_text(SOURCE, encoding="utf-8")
    page_path = ws.pages / "concepts" / "example.md"
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_path.write_text(PAGE_TEMPLATE, encoding="utf-8")
    seed = SeedClaim(
        entity_id="concept.example",
        entity_type="concept",
        display_name="Example",
        claim_id="c.ex.1",
        claim_kind="definition",
        claim_text="A canonical sentence in paragraph one.",
        locator=Locator(
            locator_type="markdown_prose_v1",
            heading_path=["Methods"],
            paragraph_index=1,
            sentence_start=1,
            sentence_end=1,
        ),
        render_target=("concept/example", "claim_block.concept.example"),
    )
    ingest(
        ws,
        src,
        source_id="src.article",
        source_class="markdown_prose",
        seed_claims=[seed],
    )
    return ws, src


# ---- generator shape ----------------------------------------------------


def test_generate_canary_token_has_prefix_and_entropy() -> None:
    t1 = generate_canary_token()
    t2 = generate_canary_token()
    assert t1.startswith(GENERATED_CANARY_PREFIX)
    assert t2.startswith(GENERATED_CANARY_PREFIX)
    # Distinct invocations produce distinct tokens (overwhelming
    # probability from 128 bits of entropy).
    assert t1 != t2
    # Non-prefix content is at least 32 hex chars (128 bits).
    suffix1 = t1[len(GENERATED_CANARY_PREFIX):]
    suffix2 = t2[len(GENERATED_CANARY_PREFIX):]
    assert len(suffix1) >= 32
    assert len(suffix2) >= 32
    # Hex suffix.
    assert all(c in "0123456789abcdef" for c in suffix1)


# ---- clean workspace + flag = still passes ------------------------------


def test_generated_canary_flag_passes_on_clean_workspace(
    tmp_path: Path, deterministic_token: str
) -> None:
    ws = _seed_clean_workspace(tmp_path)
    result = lint(ws, generated_canary=True)
    assert result.passed, (result.failures, result.canary_hits)
    assert result.canary_hits == []


# ---- planted leak paths -------------------------------------------------


def test_generated_canary_flag_flags_claim_block_leak(
    tmp_path: Path, deterministic_token: str
) -> None:
    ws, _src = _seed_claim_extract_workspace(tmp_path)
    # Plant the generated token into the rendered claim-block region.
    page_path = ws.pages / "concepts" / "example.md"
    page_text = page_path.read_text(encoding="utf-8")
    poisoned = page_text.replace(
        "## Example",
        f"## Example\n\n{deterministic_token}",
    )
    page_path.write_text(poisoned, encoding="utf-8")

    result = lint(ws, generated_canary=True)
    assert not result.passed
    assert result.canary_hits, "expected generated canary leak to be flagged"
    assert any(deterministic_token in hit for hit in result.canary_hits)
    assert any("claim-block" in hit for hit in result.canary_hits)


def test_generated_canary_flag_flags_journal_leak(
    tmp_path: Path, deterministic_token: str
) -> None:
    """Generated tokens must be scanned through the same forbidden
    observation points as the fixed token. This test plants the
    generated token inside a persisted journal entry (mirroring the
    pattern exercised by `test_canary_lint_scans_persisted_invocation_logs`)
    and confirms lint flags it."""
    ws, _src = _seed_claim_extract_workspace(tmp_path)
    journal = OperationJournal(ws)
    latest = journal.latest()
    assert latest is not None
    latest.notes.append(f"poisoned: {deterministic_token}")
    journal.save(latest)

    result = lint(ws, generated_canary=True)
    assert not result.passed
    assert any(deterministic_token in hit for hit in result.canary_hits)
    assert any("journals" in hit for hit in result.canary_hits)


def test_without_flag_generated_token_is_not_scanned(
    tmp_path: Path, deterministic_token: str
) -> None:
    """Symmetric control: if the flag is absent, lint must NOT scan
    for the generated token, so a planted token is not a failure.
    This proves the flag actually controls the scan set and the
    default behavior (fixed-token-only) is preserved."""
    ws, _src = _seed_claim_extract_workspace(tmp_path)
    journal = OperationJournal(ws)
    latest = journal.latest()
    assert latest is not None
    latest.notes.append(f"would-only-flag-if-scanned: {deterministic_token}")
    journal.save(latest)

    result = lint(ws)  # generated_canary=False by default
    assert result.passed
    assert result.canary_hits == []


# ---- CLI plumbing -------------------------------------------------------


def test_cli_passes_generated_canary_flag_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spy on lint() to confirm the CLI flag plumbs through with
    value True when --generated-canary is passed and False when it
    is omitted."""
    ws = Workspace.init(tmp_path)
    seen: list[bool] = []

    from llloom import cli as cli_module

    real_lint = cli_module.lint

    def spy(ws_arg, *, generated_canary=False, **kwargs):  # type: ignore[no-redef]
        seen.append(generated_canary)
        return real_lint(ws_arg, generated_canary=generated_canary, **kwargs)

    monkeypatch.setattr(cli_module, "lint", spy)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli_main(["--root", str(ws.root), "lint", "--generated-canary"])
    assert rc == 0
    assert seen == [True]

    # Without the flag, lint runs with generated_canary=False.
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        rc2 = cli_main(["--root", str(ws.root), "lint"])
    assert rc2 == 0
    assert seen == [True, False]

    # Sanity: the clean-workspace JSON reports passed=true with no hits.
    payload = json.loads(buf2.getvalue())
    assert payload["canary_hits"] == []
    assert payload["failures"] == []
