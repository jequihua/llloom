"""Unit tests pinning the code-backed view_render unblocking.

Two narrow contract claims:

1. the early policy-cutoff refusal that used to short-circuit
   code-backed ``claim_extract_and_view_render`` is gone from
   ``ops.ingest`` — there is no live runtime path that returns
   ``"code-page rendering is deferred"`` as a refusal reason
2. ``llloom.pages.render.render_page_file`` (the existing variant-(B)
   renderer) remains locator-agnostic — its source does not branch on
   ``code_v1`` vs ``markdown_prose_v1``, so the new code-backed
   render path reuses the same deterministic page contract that
   narrative ingests use
"""

from __future__ import annotations

import importlib
import inspect


_PRIOR_REFUSAL_SENTINEL = (
    "code-page rendering is deferred"
)


def test_ingest_no_longer_carries_code_backed_view_render_policy_refusal() -> None:
    """A regression that re-introduced the old early-refusal sentinel
    string in ``ops.ingest`` would fail this test immediately. The
    runtime no longer refuses code-backed
    ``claim_extract_and_view_render`` at the policy cutoff; it falls
    through to the same parse/validate/verify/persist/render pipeline
    as the narrative path."""
    ingest_mod = importlib.import_module("llloom.ops.ingest")
    source = inspect.getsource(ingest_mod.ingest)
    assert _PRIOR_REFUSAL_SENTINEL not in source, (
        "The early refusal for code-backed claim_extract_and_view_render "
        "was reintroduced; this slice removed it deliberately. See "
        "review_endorser_code_backed_view_render.md (forthcoming)."
    )


def test_render_page_file_is_locator_agnostic() -> None:
    """The existing variant-(B) renderer must not branch on locator
    type. A regression that added a `code_v1` / `markdown_prose_v1`
    conditional inside ``render_page_file`` would smuggle a second
    page system in, which the milestone explicitly forbade."""
    pages_render = importlib.import_module("llloom.pages.render")
    source = inspect.getsource(pages_render.render_page_file)
    assert "code_v1" not in source
    assert "markdown_prose_v1" not in source
    assert "legal_act_v1" not in source
