"""Typed-input LLM invocation harness.

The harness has no filesystem access. Every LLM-invoking operation goes
through :class:`LLMInvoke`. Excluded content (commentary regions, spine
prose, ``index_only`` source bodies) is unreachable by construction:
callers must supply pre-serialized typed inputs and the harness has no
way to fetch anything else.

This package currently ships a :class:`NullModel` that produces no
output. It exists to exercise the input-boundary contract and the
invocation-log contract without introducing a model dependency in the
first slice. Real model bindings plug in by implementing the
:class:`ModelBackend` protocol.
"""

from llloom.llm.harness import (
    ALLOWED_OPERATIONS,
    ALLOWED_READ_CLASSES,
    ALLOWED_WRITE_KINDS,
    ClaimBlockRegion,
    ClaimRecord,
    HarnessRefusal,
    InvocationLog,
    LLMInvoke,
    ModelBackend,
    NullModel,
    OperationKind,
    SchemaDocument,
    SourceDocument,
    SourceSpan,
    StructureItemContext,
    WriteTarget,
)
from llloom.llm.output import (
    ModelOutputError,
    RawCandidate,
    parse_claim_extraction_output,
)
from llloom.llm.openai_backend import (
    OpenAIBackendError,
    OpenAIModelBackend,
)
from llloom.llm.anthropic_backend import (
    AnthropicBackendError,
    AnthropicModelBackend,
)

__all__ = [
    "ALLOWED_OPERATIONS",
    "ALLOWED_READ_CLASSES",
    "ALLOWED_WRITE_KINDS",
    "AnthropicBackendError",
    "AnthropicModelBackend",
    "ClaimBlockRegion",
    "ClaimRecord",
    "HarnessRefusal",
    "InvocationLog",
    "LLMInvoke",
    "ModelBackend",
    "ModelOutputError",
    "NullModel",
    "OpenAIBackendError",
    "OpenAIModelBackend",
    "OperationKind",
    "RawCandidate",
    "SchemaDocument",
    "SourceDocument",
    "SourceSpan",
    "StructureItemContext",
    "WriteTarget",
    "parse_claim_extraction_output",
]

