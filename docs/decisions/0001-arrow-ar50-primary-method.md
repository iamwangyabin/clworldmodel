# Decision 0001: use ARROW-50 as the primary method

- Status: accepted
- Date: 2026-08-12

## Context

The project is intended to start from the strongest canonical ARROW method,
not from DreamerV3/FIFO. The ARROW paper evaluates FIFO/LTDM buffer splits of
75/25 (AR25), 50/50 (AR50), and 25/75 (AR75) at a fixed total replay budget.
No split dominates every benchmark and metric, but AR50 is the configuration
used throughout the paper's main experiments.

## Decision

ARROW-50 is the initial primary research method and implementation base. It
uses:

- the official ARROW world model and actor-critic;
- 512 FIFO and 512 LTDM trajectory slots, each of length 512;
- equal FIFO/LTDM buffer-selection probability;
- the paper's original-order six-task Atari curriculum for the first official
  reproduction;
- the five published seeds for a reproduction claim.

AR25 and AR75 remain replay-ratio ablations. DV3/FIFO remains a matched control
for attribution and is not a prerequisite for starting the ARROW-50 pilot.

## Consequences

- GPU validation starts with an ARROW-50 pilot before the complete campaign.
- Clean project-owned components should be ported from the ARROW composition,
  with replay behavior tested independently.
- Claims about superiority still require matched DV3 results and multiple
  seeds; making ARROW primary does not relax comparison requirements.
