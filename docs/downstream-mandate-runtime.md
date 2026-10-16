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

## Discord roles in MCP session metadata (CES-1448)

For a server that forwards session metadata, each Discord turn adds two host-written
fields beside the seven existing ones, for the Mandate Edge bridge to sign:

- `com.nousresearch.hermes/role_ids`: the speaking member's role ids in the message's
  own server, comma-separated, without `@everyone` (whose id is the server's). It is
  `""` for a member with no other role, and absent in a DM.
- `com.nousresearch.hermes/parent_chat_id`: a thread message's parent channel. It is
  absent for any other message. The bridge refuses it unless it equals the session
  record's parent.

The role ids live only as long as the inbound message's `MessageEvent`; no session
source, stored origin or persisted record holds them. A turn with no fresh Discord
message behind it (a plugin injection, a background notification, any internal or
synthetic turn) sends no `role_ids`. The job-thread handoff reads the member's roles
from the server instead. Under PII redaction neither field is sent, and neither ever
reaches a subprocess's environment.

Deploy order: the Core role bound (CES-1447) and Hermes's server prefixes in the
registry (CES-1451 part 1) go live first; then this branch, whose new fields the
current bridge ignores; then the bridge that signs them (CES-1449). Design:
[Discord roles in registered servers and channels](https://linear.app/cegarza/document/discord-roles-in-registered-servers-and-channels-design-ces-1377-7bfc5139812d),
ratified 2026-10-16.
