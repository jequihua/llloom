# Denied fixture: fake credentials

## Do not ingest

This file contains a synthetic fake credential for contract testing:

FAKE_API_KEY=sk-test-do-not-use-0000000000000000

The source class ``denied_fixture`` maps to policy ``deny``. Ingest must
refuse without writing claims, hashing the source, or emitting any
LLMInvoke input.
