# Use a pinned native V3 core for the source-recipe MiniGrid pilot

Date: 2026-09-06. Scope: opt-in pilot integration only.

The user requested the Continual-Dreamer implementation's approach with V3,
and matched ARROW/RS replay, rather than another run of the old low-acquisition
MiniGrid configuration. We retain the source environment conventions, P2E
formulas and action-based cadence, while importing the already-selected NM512
V3 pin unmodified. A script-layer adapter calls native WM and actor/critic
methods; new methods do not fork the ARROW trainer or modify its vendor.

Source RS's variable episode capacity/admission logic cannot be called an
unbiased LTDM implementation. The named EpisodeSlots adaptation explicitly
bounds complete episodes and uses Algorithm R. This requires separate protocol
labeling; it does not qualify as 'only a V2→V3 change'. No old run is relabeled,
and no official or five-seed campaign is launched by default.

Shared V3+P2E in both arms, CPU uint8 episode replay, one-controller extrinsic
evaluation isolated from training, fixed action/update budgets and raw returns
are invariants. Every additional deviation and resolved default is manifested.
The first decision is whether source-style exploration can acquire DoorKey;
matching paper scores and a continual method ranking require further evidence.

Native model import and new protocol work are separate commits. Dependencies
are a separate pinned reference extra; old Atari/MiniGrid extras still require
torch 2.3.0, while core compatibility admits the native reference's 2.4.1.
Inference snapshots explicitly prohibit training resumption; a full resumable
replay/RNG/optimizer/environment checkpoint is deferred, not misrepresented.
