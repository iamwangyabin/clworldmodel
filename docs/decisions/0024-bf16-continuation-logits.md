# Decision 0024: Keep continuation logits stable under BF16

## Status

Accepted as a runtime-correctness fix for every opt-in BF16 Atari profile. It
does not change the ARROW objective, model parameters, data budget, or update
budget.

## Evidence

The stopped CNN-FullBank run at project commit
`2458a023786c104388ee9ae0e48279646924a4f2` reproduced its preceding
`x4-full-updates` loss trace exactly through world-model step 11,000. At step
10,000 its world-model losses were:

| Metric | CNN-FullBank BF16 DP4 | FP32 ARROW diagnostic | Ratio |
| --- | ---: | ---: | ---: |
| reconstruction | 6.1612 | 7.9063 | 0.78 |
| KL | 2.6194 | 3.0570 | 0.86 |
| reward | 0.00592 | 0.01395 | 0.42 |
| continuation | 0.24414 | 0.00849 | 28.77 |

The other losses were finite and of the same order. Continuation loss alone
appeared in exact multiples of `0.048828125`, and the terminal-frame continue
probability reached exactly `1.0`.

The cause was the terminal `Sigmoid` inside the continuation MLP. CUDA BF16
autocast rounded sufficiently positive logits to exactly one. Converting that
already rounded probability to float32 before binary cross entropy was too
late; each contradicting terminal target then received the framework's clamped
loss of 100.

The FP32 ARROW run is a numerical diagnostic rather than a paired performance
control: it used one device, float32 compute and replay, a fixed global batch,
and the older advancing evaluation stream. Those differences cannot explain
the exact BF16 saturation signature.

## Decision

- Remove the terminal sigmoid module and expose continuation logits.
- Compute `sigmoid(logits.float())` with autocast disabled.
- Train with float32 `binary_cross_entropy_with_logits`.
- Supply the same float32 probability to imagined rollouts and evaluation.
- Preserve all old state-dict tensor keys and parameter shapes.

The affected 180-epoch run was stopped after epoch 13. Its epoch-10 inference
snapshot, logs, TensorBoard data, run manifest, and explicit stop record remain
preserved. It is non-resumable and must not be presented as a completed result.

## Validation

A focused mixed-precision test forces a continuation logit of 10 under BF16
autocast. It requires a probability strictly below one, a logit-space loss near
10 rather than 100, and a finite nonzero gradient. A fresh GPU smoke must also
confirm that early continuation loss no longer follows the clamped quantized
trace before another acquisition run proceeds.
