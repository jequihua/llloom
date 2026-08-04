# Concurrency fixture

This directory holds no source text. The concurrency scenario is built
programmatically in ``tests/integration/test_concurrency.py``:

1. acquire the workspace lock with a first simulated op,
2. attempt ``ingest`` in a second op and assert it refuses,
3. make the first op appear stale by rewriting its heartbeat into the
   past (or by journal-backed forensic state),
4. run ``reconcile`` and assert the stale lock clears and the journal
   entry is marked ``interrupted``,
5. retry the second ingest and assert success.

No raw source is required; the concurrency scenario operates on the
fixture corpus already registered by earlier steps.
