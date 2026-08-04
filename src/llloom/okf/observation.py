"""Bounded, read-only OKF / framework-profile observation for one path.

This is llloom's owned consumer boundary for the pinned candidate profile
``0.1-rc.1``. It reproduces the accepted ``full_parser`` outcomes recorded in
``02_analysis/llloom_okf_consumer_contract_and_fixture_plan.md`` and adds no
authority: ``execution_eligibility`` is the constant ``not_evaluated`` and no
result here may reach claim verification, lifecycle, source status, render
eligibility, query inclusion, model permissions, MCP mutation, or operation
execution.

The path is observed, never touched: no write, lock, journal, transaction,
sidecar, report, model or provider call, network access, mtime change, or input
mutation. The file is opened once, read up to a fixed ceiling plus one byte, and
decoded once.

PyYAML supplies YAML syntax and representation only, through the pure-Python
``yaml.SafeLoader``. Every producer policy -- Markdown framing, resource bounds,
semantic duplicate-key rejection, canonical scalar rules, and the separated layer
results -- lives here. There is no fallback engine: a missing PyYAML is an
installation failure, never different semantics.

Marker health is deliberately absent. Region and marker structure stay owned by
``llloom.pages.regions`` and the render plan; this module never calls them, so a
caller cannot mistake one result for a summary of both layers.
"""

from __future__ import annotations

import re
import stat as stat_module
from dataclasses import dataclass
from pathlib import Path

import yaml  # mandatory dependency; a missing install surfaces as ImportError

__all__ = ["PageOkfObservation", "observe_page_okf"]


PINNED_FRAMEWORK_PROFILE = "0.1-rc.1"

# Shared type registry of the pinned candidate profile, including the tool-owned
# reserved types. Membership decides PROFILE_TYPE_UNSUPPORTED only; it never
# gates OKF concept validity.
PROFILE_TYPE_REGISTRY = frozenset({
    "brief", "constraint", "decision", "analysis", "coding_prompt", "review_prompt",
    "self_report", "review_report", "verdict_record", "delivery_plan", "framework_doc",
    "source", "claim", "entity", "page", "milestone", "slice",
})

# Total-input ceiling, enforced on raw bytes before decode or any scan.
MAX_ARTIFACT_BYTES = 1_048_576
# Frontmatter-block ceilings, enforced before PyYAML sees anything.
MAX_FRONTMATTER_BYTES = 65_536
MAX_FRONTMATTER_LINES = 500
MAX_LINE_LEN = 8_192
# Parse and graph ceilings.
MAX_TOKENS = 10_000
MAX_NODES = 2_000
MAX_DEPTH = 32
MAX_SCALAR_LEN = 16_384
MAX_MAPPING_ITEMS = 500
MAX_SEQUENCE_ITEMS = 1_000
MAX_ALIASES = 50

_INT_CANONICAL_RE = re.compile(r"^(?:0|-?[1-9][0-9]*)$")
# A plain lexeme that looks numeric but that PyYAML left as a string is a
# cross-parser hazard, so it is outside the producer subset.
_NUMERIC_LIKE_RE = re.compile(r"^[+-]?[0-9][0-9_]*([.:][0-9_]+)*([eE][+-]?[0-9]+)?$")

_STR_TAG = "tag:yaml.org,2002:str"
_INT_TAG = "tag:yaml.org,2002:int"
_BOOL_TAG = "tag:yaml.org,2002:bool"
_NULL_TAG = "tag:yaml.org,2002:null"
_MERGE_TAG = "tag:yaml.org,2002:merge"

_DELIMITER = "---"


@dataclass(frozen=True)
class PageOkfObservation:
    """One read-only observation of a path's OKF and framework-profile status.

    The three layers stay separated and are never inferred from one another.
    ``read_status`` is llloom-owned and carries no OKF or profile semantics; it
    exists so that an absent target, an oversize target, an undecodable target,
    and a decoded target without frontmatter remain four distinct answers.

    ``declared_framework_profile`` echoes the declared version lexeme exactly as
    written: the scalar's **source content** with only its matching syntactic quote
    delimiters stripped, before YAML escape normalization. A source spelling of
    ``"0.1\\x2drc.1"`` is therefore returned with its backslash intact and does not
    match the candidate, even though YAML would decode it to ``0.1-rc.1``. It is
    populated when a declared version was actually read: for a supported version and
    for an unsupported one.
    """

    read_status: str
    okf_concept_result: str
    okf_concept_reason: str | None
    framework_profile_result: str
    framework_profile_reason: str | None
    execution_eligibility: str
    declared_framework_profile: str | None


class _ResourceRefusal(Exception):
    """A finite parse bound was exceeded. Not proof of invalid YAML."""


def _observation(
    read_status: str,
    okf_result: str,
    okf_reason: str | None,
    profile_result: str,
    profile_reason: str | None,
    declared: str | None = None,
) -> PageOkfObservation:
    return PageOkfObservation(
        read_status=read_status,
        okf_concept_result=okf_result,
        okf_concept_reason=okf_reason,
        framework_profile_result=profile_result,
        framework_profile_reason=profile_reason,
        execution_eligibility="not_evaluated",
        declared_framework_profile=declared,
    )


def _limit_exceeded(read_status: str) -> PageOkfObservation:
    """A bounded refusal: OKF ``unverified``, never ``fail``."""
    return _observation(
        read_status, "unverified", "OKF_PARSE_LIMIT_EXCEEDED",
        "fail", "PROFILE_YAML_OUT_OF_SUBSET",
    )


@dataclass(frozen=True)
class _ReadSnapshot:
    """One immutable bounded read of a target: the only input the parse core sees.

    ``status`` is ``ok``, ``target_absent``, ``unreadable``, ``oversize``, or
    ``undecodable``. ``text`` is the decoded content, and only when ``status`` is
    ``ok``. Once this exists the parse is fixed: nothing the filesystem does
    afterwards can change the observation computed from it.
    """

    status: str
    text: str | None = None


def _snapshot_from_bytes(data: bytes) -> _ReadSnapshot:
    """Apply the total-input ceiling and the single decode to bytes already read."""
    if len(data) > MAX_ARTIFACT_BYTES:
        return _ReadSnapshot("oversize")
    try:
        return _ReadSnapshot("ok", data.decode("utf-8"))
    except UnicodeDecodeError:
        return _ReadSnapshot("undecodable")


def _read_snapshot(path: Path) -> _ReadSnapshot:
    """One stat, one open, one bounded read, one decode.

    The sequence is the one authorized by the M002 change-control amendment A1, with
    the invalid-spelling rule of addendum B1: stat first; absence and
    not-a-directory to R0; an unusable path spelling that the runtime cannot submit
    to filesystem metadata to R0; any other metadata ``OSError`` to R0U; a
    non-regular result to R0; then one open and one ``read(ceiling + 1)``, with an
    absence race to R0 and any other error to R0U.

    ``ValueError`` is caught only here, at the path and filesystem boundary. A
    ``TypeError`` from outside the declared input domain, an exception from a broken
    custom ``__fspath__``, ``MemoryError``, and ``KeyboardInterrupt`` are not caught
    and are not mapped to any row.
    """
    try:
        info = path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return _ReadSnapshot("target_absent")
    except ValueError:
        # An embedded NUL or an overlong Windows path never resolves to a target.
        return _ReadSnapshot("target_absent")
    except OSError:
        return _ReadSnapshot("unreadable")
    if not stat_module.S_ISREG(info.st_mode):
        return _ReadSnapshot("target_absent")
    try:
        with open(path, "rb") as handle:
            data = handle.read(MAX_ARTIFACT_BYTES + 1)
    except (FileNotFoundError, NotADirectoryError):
        return _ReadSnapshot("target_absent")   # the target raced away after stat
    except ValueError:
        return _ReadSnapshot("target_absent")
    except OSError:
        return _ReadSnapshot("unreadable")
    return _snapshot_from_bytes(data)


@dataclass(frozen=True)
class _FramedBlock:
    """The exact source slice between the frontmatter delimiters, plus its shape.

    ``raw`` is the byte-faithful source content: interior LF and CRLF sequences are
    preserved, and neither the ending after the opening delimiter nor the ending
    before the closing delimiter is included. Bounds and lexemes are both measured
    against this one slice, so they cannot drift apart.

    ``raw`` is materialized only for a block that is still within the character span
    a passing block could have. A block already known to be over a bound carries
    ``over_bound`` instead, so a hostile input is never copied out of the source.
    """

    status: str                  # "ok" | "legacy" | "unterminated"
    raw: str = ""
    line_count: int = 0
    longest_line: int = 0
    over_bound: bool = False


def _iter_lines(text: str):
    """Yield ``(content, start, content_end, next_start)`` for each LF/CRLF line.

    Only LF and CRLF terminate a line. Bare CR, VT, FF, NEL, and the Unicode line
    and paragraph separators are ordinary content, so they can never form a
    delimiter -- which is exactly what ``str.splitlines`` got wrong here.
    """
    position = 0
    length = len(text)
    while position < length:
        newline = text.find("\n", position)
        if newline == -1:
            yield text[position:length], position, length, length
            return
        content_end = newline
        if content_end > position and text[content_end - 1] == "\r":
            content_end -= 1
        yield text[position:content_end], position, content_end, newline + 1
        position = newline + 1


def _frame_frontmatter(text: str) -> _FramedBlock:
    """Frame the frontmatter block under llloom's own line-terminator contract.

    Streaming and bounded: the line iterator is consumed lazily, no line is ever
    stored, and only integer counters and offsets are retained. A legacy input
    returns after one line. An input that has already breached a bound keeps
    scanning without copying anything, purely to decide whether a closing delimiter
    exists -- which is what separates a terminated over-bound block (R5) from an
    unterminated one (R4). Framing therefore still precedes bounds.

    Deliberately independent of ``llloom.pages.regions``: the native region parser
    requires LF and a trailing newline, and reusing it here would make the OKF
    boundary inherit a framing rule it does not share.
    """
    lines = _iter_lines(text)
    first = next(lines, None)
    if first is None or first[0] != _DELIMITER:
        return _FramedBlock("legacy")

    content_start = first[3]
    line_count = 0
    longest_line = 0
    last_content_end = content_start
    over_bound = False

    for content, _start, content_end, _next_start in lines:
        if content == _DELIMITER:
            if line_count == 0:
                return _FramedBlock("ok")
            if over_bound:
                return _FramedBlock("ok", "", line_count, longest_line, True)
            raw = text[content_start:last_content_end]
            # Character span already bounds the byte span from below, so this
            # encode only ever runs on a block that could still pass.
            if len(raw.encode("utf-8", "surrogatepass")) > MAX_FRONTMATTER_BYTES:
                return _FramedBlock("ok", "", line_count, longest_line, True)
            return _FramedBlock("ok", raw, line_count, longest_line, False)

        line_count += 1
        last_content_end = content_end
        if len(content) > longest_line:
            longest_line = len(content)
        if not over_bound and (
            line_count > MAX_FRONTMATTER_LINES
            or longest_line > MAX_LINE_LEN
            # One character is at least one UTF-8 byte, so a character span past the
            # ceiling is a byte span past the ceiling. Checking it here keeps the
            # slice-and-encode above off the hostile path entirely.
            or last_content_end - content_start > MAX_FRONTMATTER_BYTES
        ):
            over_bound = True
    return _FramedBlock("unterminated")


def _within_bounds(block: _FramedBlock) -> bool:
    """Whether the framed block passed every pre-parse ceiling.

    The byte ceiling counts the raw UTF-8 bytes, so a CRLF document is measured by
    the bytes it actually contains rather than by an LF-normalized reconstruction.
    The measurement happens during framing; this only reports it.
    """
    return not (
        block.over_bound
        or block.line_count > MAX_FRONTMATTER_LINES
        or block.longest_line > MAX_LINE_LEN
    )


def _scan_features(frontmatter: str) -> set[str]:
    """Collect producer-subset feature flags and enforce token and alias bounds.

    Raises ``yaml.YAMLError`` on a syntax defect, ``RecursionError`` on
    pathological input, and ``_ResourceRefusal`` on a bound.
    """
    features: set[str] = set()
    tokens = 0
    aliases = 0
    for token in yaml.scan(frontmatter, Loader=yaml.SafeLoader):
        tokens += 1
        if tokens > MAX_TOKENS:
            raise _ResourceRefusal("token count")
        if isinstance(token, (yaml.AnchorToken, yaml.AliasToken)):
            features.add("anchor_alias")
            if isinstance(token, yaml.AliasToken):
                aliases += 1
                if aliases > MAX_ALIASES:
                    raise _ResourceRefusal("alias count")
        elif isinstance(token, yaml.TagToken):
            features.add("explicit_tag")
        elif isinstance(token, (yaml.FlowMappingStartToken, yaml.FlowSequenceStartToken)):
            features.add("flow")
        elif isinstance(token, yaml.ScalarToken):
            if token.style == "'":
                features.add("single_quote")
            elif token.style in ("|", ">"):
                features.add("block_scalar")
    return features


def _canonical_scalar_ok(node: yaml.ScalarNode) -> bool:
    """Whether a scalar value obeys the canonical producer subset.

    Judged from PyYAML's resolved tag plus the original lexeme, never from the
    constructed Python value, so ``01`` and ``0`` stay distinguishable.
    """
    style = node.style
    if style == '"':
        return True                       # a double-quoted string is canonical
    if style in ("'", "|", ">"):
        return False                      # single-quote and block scalars are out
    lexeme = node.value
    if node.tag == _STR_TAG:
        return not _NUMERIC_LIKE_RE.match(lexeme)
    if node.tag == _INT_TAG:
        return bool(_INT_CANONICAL_RE.match(lexeme))
    if node.tag == _BOOL_TAG:
        return lexeme in ("true", "false")
    if node.tag == _NULL_TAG:
        return lexeme in ("null", "~", "")
    return False                          # float, timestamp, or any other tag


def _key_identity(loader: yaml.SafeLoader, key_node) -> tuple[object | None, bool]:
    """Semantic identity of a mapping key, and whether it is a string scalar.

    The identity is ``(resolved_tag, constructed_scalar_value)``, so ``1``/``01``,
    ``60``/``1:0``, ``yes``/``on``/``true``, and ``null``/``~`` collapse to one
    key while integer ``1`` and string ``"1"`` stay distinct. Returns
    ``(None, False)`` for a complex or unconstructable key, where equality cannot
    be established safely.
    """
    if not isinstance(key_node, yaml.ScalarNode):
        return None, False
    tag = key_node.tag
    try:
        value = loader.construct_object(key_node, deep=True)
    except (yaml.YAMLError, RecursionError, ValueError, TypeError):
        return None, False
    try:
        identity = (tag, value)
        hash(identity)
    except TypeError:
        identity = (tag, repr(value))
    return identity, tag == _STR_TAG


def _walk(node, depth: int, loader: yaml.SafeLoader, state: dict) -> None:
    """Bounded, cycle-safe traversal collecting subset and duplicate evidence.

    Unique nodes are counted by identity, so a shared alias target is inspected
    once; the active stack detects an alias cycle before it can be followed.
    """
    if depth > MAX_DEPTH:
        raise _ResourceRefusal("nesting depth")
    node_id = id(node)
    if node_id in state["active"]:
        raise _ResourceRefusal("alias cycle")
    if node_id in state["visited"]:
        return
    state["visited"].add(node_id)
    if len(state["visited"]) > MAX_NODES:
        raise _ResourceRefusal("node count")

    if isinstance(node, yaml.ScalarNode):
        if len(node.value) > MAX_SCALAR_LEN:
            raise _ResourceRefusal("scalar length")
        return

    state["active"].add(node_id)
    if isinstance(node, yaml.MappingNode):
        if node.flow_style:
            state["features"].add("flow")
        if len(node.value) > MAX_MAPPING_ITEMS:
            raise _ResourceRefusal("mapping size")
        seen: set[object] = set()
        for key_node, value_node in node.value:
            if isinstance(key_node, yaml.ScalarNode) and key_node.tag == _MERGE_TAG:
                # A merge key is both a producer-subset feature and a duplicate
                # candidate: its identity is a sentinel distinct from a quoted
                # "<<" string key, so two merge keys in one mapping duplicate.
                state["features"].add("merge")
                identity = (_MERGE_TAG, None)
                if identity in seen:
                    state["duplicate"] = True
                seen.add(identity)
            else:
                identity, is_string = _key_identity(loader, key_node)
                if identity is None:
                    state["complex_key"] = True
                else:
                    if identity in seen:
                        state["duplicate"] = True
                    seen.add(identity)
                    if not is_string:
                        state["nonstring_key"] = True
            _walk(key_node, depth + 1, loader, state)
            _walk(value_node, depth + 1, loader, state)
            if isinstance(value_node, yaml.ScalarNode) and not _canonical_scalar_ok(value_node):
                state["noncanonical"] = True
    elif isinstance(node, yaml.SequenceNode):
        if node.flow_style:
            state["features"].add("flow")
        if len(node.value) > MAX_SEQUENCE_ITEMS:
            raise _ResourceRefusal("sequence size")
        for item in node.value:
            _walk(item, depth + 1, loader, state)
            if isinstance(item, yaml.ScalarNode) and not _canonical_scalar_ok(item):
                state["noncanonical"] = True
    state["active"].discard(node_id)


def _top_level_scalar(root, key: str) -> tuple[bool, yaml.ScalarNode | None]:
    """Return ``(present, value_node)`` for a top-level string-keyed scalar field.

    The node is returned rather than its decoded value so a caller can choose
    between the semantic value and the exact source lexeme.
    """
    if not isinstance(root, yaml.MappingNode):
        return False, None
    for key_node, value_node in root.value:
        if (
            isinstance(key_node, yaml.ScalarNode)
            and key_node.tag == _STR_TAG
            and key_node.value == key
        ):
            if isinstance(value_node, yaml.ScalarNode):
                return True, value_node
            return True, None
    return False, None


def _source_lexeme(source: str, node: yaml.ScalarNode) -> str:
    """The scalar's source content, before YAML escape normalization.

    Only the matching syntactic quote delimiters are stripped. Nothing is
    unescaped, normalized, or reconstructed, so ``"0.1\\x2drc.1"`` in the file is
    returned with its backslash intact and does not equal the pinned candidate.
    """
    text = source[node.start_mark.index:node.end_mark.index]
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        return text[1:-1]
    return text


def observe_page_okf(path: Path | str) -> PageOkfObservation:
    """Observe one path's OKF concept and framework-profile status, read-only.

    The input domain is any path. Absent, non-regular, unusable, unreadable,
    oversize, undecodable, legacy, and malformed targets all have a defined answer,
    so a caller never needs a precondition and never sees an exception from an
    ordinary input.
    """
    return _observe_snapshot(_read_snapshot(Path(path)))


def _observe_snapshot(snapshot: _ReadSnapshot) -> PageOkfObservation:
    """The parse core. Every observation, public or provenance-gated, ends here.

    Taking an immutable snapshot rather than a path is what makes the fixture
    parity gate honest: the bytes whose digest was verified are the bytes parsed,
    with no interval in which the filesystem could substitute different ones.
    """
    text, read_status = snapshot.text, snapshot.status

    # 1. Results owned by the bounded read (rows R0, R0U, R1, R2).
    if read_status in ("target_absent", "unreadable"):
        return _observation(read_status, "not_evaluated", None, "not_applicable", None)
    if read_status == "oversize":
        return _limit_exceeded("oversize")
    if read_status == "undecodable":
        return _observation("undecodable", "not_evaluated", None, "not_applicable", None)
    assert text is not None

    # 2. Markdown framing, owned here (rows R3, R4).
    block = _frame_frontmatter(text)
    if block.status == "legacy":
        return _observation("ok", "not_evaluated", None, "not_applicable", None)
    if block.status == "unterminated":
        return _observation("ok", "fail", "OKF_FRONTMATTER_MISSING", "not_applicable", None)
    frontmatter = block.raw

    # 3. Raw frontmatter bounds, before PyYAML constructs anything (row R5).
    if not _within_bounds(block):
        return _limit_exceeded("ok")

    # 4. Syntax scan, feature detection, token and alias bounds (rows R6, R7).
    try:
        features = _scan_features(frontmatter)
    except (_ResourceRefusal, RecursionError):
        return _limit_exceeded("ok")
    except yaml.YAMLError:
        return _observation(
            "ok", "fail", "OKF_YAML_INVALID", "fail", "PROFILE_YAML_OUT_OF_SUBSET"
        )

    # 5. Compose exactly one document graph, then traverse it under bounds.
    loader = yaml.SafeLoader(frontmatter)
    try:
        try:
            root = loader.get_single_node()   # raises on more than one document
        except yaml.YAMLError:
            return _observation(
                "ok", "fail", "OKF_YAML_INVALID", "fail", "PROFILE_YAML_OUT_OF_SUBSET"
            )
        except RecursionError:
            return _limit_exceeded("ok")

        state = {
            "active": set(), "visited": set(), "features": set(features),
            "duplicate": False, "complex_key": False, "nonstring_key": False,
            "noncanonical": False,
        }
        if root is not None:
            try:
                _walk(root, 0, loader, state)
            except (_ResourceRefusal, RecursionError):
                return _limit_exceeded("ok")

        # 6. A semantic duplicate key is a conclusive YAML failure (row R8).
        if state["duplicate"]:
            return _observation(
                "ok", "fail", "OKF_YAML_INVALID", "fail", "PROFILE_YAML_OUT_OF_SUBSET"
            )

        # 7. OKF concept: a non-empty top-level string-keyed `type` (row R9).
        present, type_node = _top_level_scalar(root, "type")
        type_value = type_node.value if type_node is not None else None
        if not present or not type_value:
            return _observation("ok", "fail", "OKF_TYPE_MISSING", "not_applicable", None)

        # 8. Framework profile, evaluated independently of the OKF pass.
        out_of_subset = (
            bool(state["features"]) or state["noncanonical"]
            or state["complex_key"] or state["nonstring_key"]
        )
        if out_of_subset:                                                    # R10
            return _observation("ok", "pass", None, "fail", "PROFILE_YAML_OUT_OF_SUBSET")
        if type_value not in PROFILE_TYPE_REGISTRY:                          # R11
            return _observation("ok", "pass", None, "fail", "PROFILE_TYPE_UNSUPPORTED")
        present_profile, profile_node = _top_level_scalar(root, "framework_profile")
        if not present_profile or profile_node is None:                      # R12
            return _observation("ok", "pass", None, "not_applicable", None)
        # The exact source lexeme, not the YAML-decoded value: both the echoed
        # field and the candidate comparison use it.
        declared = _source_lexeme(frontmatter, profile_node)
        if declared != PINNED_FRAMEWORK_PROFILE:                             # R13
            return _observation(
                "ok", "pass", None, "fail", "PROFILE_VERSION_UNSUPPORTED", declared
            )
        return _observation("ok", "pass", None, "pass", None, declared)      # R14
    finally:
        loader.dispose()
