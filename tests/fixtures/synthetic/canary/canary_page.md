---
page_id: canary/leak_test
page_class: concept
write_policy: mixed
status: rendered
---

<!-- llloom:claim-block id=claim_block.canary.leak_test -->
## Canary leak test

This claim block is populated by the renderer from the canary fixture
entity's assertions during the contract test.
<!-- /llloom:claim-block -->

<!-- llloom:commentary id=commentary.canary.leak_test owner=human -->
Commentary region seeded with the fixed fixture canary token.
If any authoritative operation ever produces this token downstream, the
exclusion contract is broken:

LLLOOM_CANARY_FIXED_Z9F3

The lint canary check must flag any leakage of this token into claim
records, rendered claim blocks, query answers, or LLMInvoke output
payloads.
<!-- /llloom:commentary -->

