# Downstream Mandate runtime

`downstream/mandate-runtime` is the deployment branch for the Mandate completion
integration in the `cesaregarza/hermes-agent` fork. It is intentionally separate
from `codex/ces-400-mcp-session-meta`, which is the head of upstream
NousResearch/hermes-agent PR #59081.

Never merge or push downstream runtime changes into that upstream PR branch.
Prepare release changes on separate branches, validate them with the
`Plugin completion contract` workflow, then merge into `downstream/mandate-runtime`.
Deploy an exact commit, retaining the previous release for rollback.

The isolated branch starts at deployed commit
`646416f42e14d801753af33832c89d0853250692`. The upstream PR must remain at that
commit throughout this release. Verify its `headRefOid` before publication and
after the downstream merge. Future upstream-only work may advance that PR
independently; downstream changes must not ride along.

The completion workflow has read-only repository permissions and does not push
source, open upstream PRs or merge upstream branches. There is no automatic
upstream synchronization configured for this downstream release path.
