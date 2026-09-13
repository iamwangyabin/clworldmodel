# AWM-AutoRoute: five fresh Atari pilot seeds after host reboot

> Historical campaign record, not the active paper backlog. Current paper
> scope and progress are defined only in
> [`AWM_AUTOROUTE_PAPER_PLAN.md`](AWM_AUTOROUTE_PAPER_PLAN.md). The states below
> are timestamped observations and must not be read as current process status.

## Request and protocol

On 2026-09-10 the user requested fresh seeds after rebooting 4090-1, 4090-2,
and 3090, then confirmed the proposed 2/2/1 allocation. These are new runs,
not resumptions of S6–S8. All previous completed, stopped, and failed runs
and their consumed-work records remain preserved.

Use the sole retained **AWM-AutoRoute** implementation, protocol v4:
two-frame cumulative probability reconstruction with unchanged-route policy
state preserved. The original six Atari tasks, order, 90 epochs per task
(540 total), training/evaluation/replay budgets, BF16 configuration, and
eight CPU threads per run are unchanged. Classification remains **pilot**.

The exact source is the already-pushed explicit-seed revision
`8f39d00dfce8611956c37b25b0b2219f81347e2c`, branch
`codex/awm-custom-seeds-20260909`. The controller fetched the GitHub upstream
and verified zero ahead/behind; targets fetched the matching hash-verified
bundle and used clean source checkouts. No research-code change was made.

## Predeclared allocation

Five distinct `secrets.randbits(32)` values were drawn once before seeing any
new results, excluding all seeds in the preceding campaign. No baseline
seed preset was changed, and no new seed was redrawn based on results.

| Label | Actual seed | Host | Deployment state at 05:41 UTC |
| --- | ---: | --- | --- |
| S9 | 24777119 | 4090-1 | Started from scratch |
| S10 | 332571952 | 4090-1 | Started from scratch |
| S11 | 2372752314 | 4090-2 | Started from scratch after verified cleanup |
| S12 | 3584376502 | 4090-2 | Started from scratch after verified cleanup |
| S13 | 2530744885 | 3090 | Started from scratch |

The two 4090-1 jobs share a GPU and CPU; 3090 runs one job because its 16 GB
RAM does not satisfy the existing two-job preflight margin. This is not a
hardware-controlled throughput comparison. 4090-2 has about 35 GiB free,
below the preserved 96 GiB two-job reservation. Its jobs must not be launched
by weakening the resource check. The user subsequently explicitly authorized
disk cleanup; the verified same-server archival below cleared this blocker.

## Initial execution evidence

- 4090-1 reports BIOS 1836 after reboot. This does **not** establish that the
  historical CPU/platform instability has been repaired.
- Both launched hosts passed the 77 retained-method tests and the target
  CUDA smoke in their current boots. Runtime packages match the original
  campaign reference.
- S9, S10, and S13 were launched around 05:19 UTC on 2026-09-10. Initial
  acceptance verified live trainer processes, the actual seed, clean synced
  provenance, routing version 4, and `from_scratch=true`.
- Their actual resolved configs differ from the preserved S0 reference only
  in `seed`. No weights, replay, RNG state, or counters were inherited.
- Initial snapshots contained no training traceback or matched current-boot
  critical kernel error. A successful startup is not a performance result
  or a long-term hardware stability test.
- No automatic resume, retry, or scheduled monitoring was enabled.
- After cleanup, 4090-2 passed the same 77 tests and CUDA smoke. S11/S12
  started at approximately 05:39 UTC. At 05:41 UTC all five actual trainer
  processes were verified running with the intended seeds and protocol:
  S9/S10 had completed seven epochs each, S13 ten, and S11/S12 were in their
  first epoch. These are startup/progress observations, not outcome claims.

Ignored execution evidence is under
`runs/awm_autoroute_atari_5seed_20260910/`: immutable `request.json`, current
`campaign.json`, upstream verification and bundle, source-config reference,
both hosts' test/smoke logs, per-run deployment records, and acceptance
snapshots. The launch and inspection controllers are preserved there too.
The preceding campaign and S0 complete backup remain separate.

## Authorized 4090-2 disk cleanup

After the user's explicit cleanup request, 26 inactive Replay arrays in three
legacy experiment directories were archived and verified on **4090-2 itself**.
The initially proposed complete export to the Mac was blocked by automatic
safety review and was not executed. The alternative sends no model, Replay,
or result contents off the server; only inventories and verification records
are retained locally.

Every archived array's SHA-256 matched its source. The compressed archive is
8,232,363,705 bytes, SHA-256
`4d3f8e53e0c103dc9fa27337eff81cc75cd975c09daaea8a210e3c76a88c50c9`.
After rechecking archive integrity, source identities, and no open users, only
those 26 `.uint8.mmap` arrays were removed. All 3,015 other regular files in
the inventoried directories remain, including models, logs, raw metrics, and
source code. Net space recovered was about 68.46 GiB; user-available space
became 103.62 GiB. The existing 96 GiB two-job check was not weakened.

The archive was initially stored with its source manifest, verification, and
cleanup record in the 4090-2 campaign's
`attempts/replay_archive_for_seeds11_12_20260910/` directory. The archive itself
was subsequently deleted by explicit user request, as recorded below; the
manifests and verification history remain.
Each affected legacy directory contains `REPLAY_ARCHIVE_RESTORE_20260910.json`.
Restore and SHA-256-verify the recorded arrays before any historical resume;
the current on-disk legacy checkpoints must not be described as immediately
resumable without this restoration. No separate off-machine Replay backup was
created in this operation. Local cleanup evidence is in the new campaign's
`4090-2-cleanup/` directory.

## Subsequent explicit deletion of Replay archives

After receiving the inventory of five Replay archives, the user explicitly
requested their deletion. The agent disclosed before deletion that the latest
combined archive had no known Mac backup and that its removal would eliminate
the retained recovery source for those legacy Replay arrays. Exactly those
five inactive Replay compressed files were removed, releasing about 11.23 GiB.
The six environment/configuration/validation packages were not removed.
Models, raw scores, logs, source, and current S9–S13 runs were not deletion
targets. File identities and lack of open users were checked before removal.

**The latest legacy Replay archive no longer exists.** The three affected
legacy restore notes now mark the archive unavailable. Earlier instructions
to extract that archive are historical, not currently executable recovery
instructions. A full historical checkpoint resume now requires an independent
exact Replay backup; no such off-machine backup was created in this operation.
Raw metrics and failure/provenance records remain available for reporting.

Remote deletion evidence is retained in
`attempts/replay_archives_deleted_by_user_20260910/` within the 4090-2 campaign;
archive directories have explicit deletion notices. Local evidence and current
availability are recorded in `4090-2-cleanup/archive-deletion-result.json` and
the current `campaign.json`.

## Interpretation boundary

The broader sequence of seed replacements was requested after observing
mixed/poor results and host failures. Preserving S0 plus fresh replacements
does not create an unbiased preregistered multi-seed estimate. Report earlier
stops/failures and consumed work alongside any eventual aggregate; do not
silently exclude them to improve the apparent method performance.
