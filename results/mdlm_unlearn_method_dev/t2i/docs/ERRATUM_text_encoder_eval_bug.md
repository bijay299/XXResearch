# ERRATUM — post-attack evaluation loaded the BASE text encoder; all text-encoder-arm results are VOID

**Date:** 2026-10-23 · **Severity: CRITICAL.** This invalidates the headline numbers of
`FINDING34`, `FINDING37`, and the text-encoder rows of `FINDING35`, plus every composite arm.

---

## 1. The bug

`attack_relearn.py` measured the pre-attack state with the erased text encoder and the post-attack
state **without it**:

```python
half  = build_pipe(torch.float16, a.unet, dev, te_path=a.text_encoder)   # BEFORE — correct
after = build_pipe(torch.float16, f"{tmp}/unet.pt", dev)                 # AFTER  — te_path MISSING
```

Official AdvUnlearn erases the **text encoder**, not the UNet. So for every arm passing
`--text-encoder`, `T_after` and `H_after` were read off a pipeline whose text encoder had **never
been erased** — i.e. the base model. Those numbers could only ever return the base model's value,
and the base model scores 1.000 on every concept (`gen_base_sd14`).

## 2. How it was caught

Not by review — by a number being too good. `H58` reported that **2 gradient steps on 1 image**
fully restored both concepts. Two steps cannot move an 859M UNet that far, so the measurement was
checked instead of the result being written up. The symptoms were all present and I had read past
them:

- the attacker-cost curve (`FINDING37`) was **perfectly flat** across a 55× range of actual work
- `T_after` and `H_after` were **identical to three decimals at every budget** (church 1.000 /
  1.000; Van Gogh 1.000 / 1.000)
- `T_before`/`H_before` varied sensibly between arms while the "after" values never varied at all

A flat curve with zero variance is the signature of a measurement that is not reading the thing
being varied. I noted the flatness twice and treated it as a finding rather than a symptom.

## 3. What is VOID

Every arm attacked with `--text-encoder`:

| document | affected claim |
|---|---|
| `FINDING34` | "official robust SOTA fully reversed, 6/6" — **void** |
| `FINDING35` §1a, §1 (AdvUnlearn/composite rows) | the 1.000 values for text-encoder arms — **void** |
| `FINDING37` | the entire attacker-cost curve, incl. "1 image / 50 steps" — **void** |
| H46, H49, H52, H53, H54, H55, H57, H58 | all post-attack numbers — **void** |
| "21/21 and 22/22 attacks at 1.000" | **void** |

## 4. What is NOT affected

Arms whose erasure lives in the **UNet**, attacked with `--unet` only. There the attacked UNet was
loaded correctly and the base text encoder is the right one:

- UCE and ESD reversals (`T2I_REPORT` §2(i), the 6/6 figure)
- the damage+distil ladders from the **UCE teacher** (`FINDING32`, `FINDING35` lower rows)
- `H45`'s utility-matched comparison
- `H51`'s forget-Fisher arms (retain 0.0917 / 0.050 — utility numbers, unaffected)
- `FINDING36`'s loss-probe result: its probe is computed on the attacker's own latents inside the
  training loop and never touches the evaluation pipeline. The base-vs-erased comparison that
  carries it is also unaffected.
- every **pre-attack** erasure measurement (`gen_*`), which used `gen_eval.py` with the text
  encoder correctly loaded

## 5. The fix, and what is being re-run

`after = build_pipe(..., te_path=te_after)`, where `te_after` is the erased encoder — or the
attacked copy of it when `--attack-te` was used, which is now persisted rather than discarded.

`H59` is re-measuring church and Van Gogh at seeds {0,1,6} × 300 steps plus the 50- and 2-step
points, with the corrected harness. **No claim about text-encoder arms stands until it reports.**

## 6. The lesson, stated against myself

This is the fifth time in this project a too-good number has turned out to be an instrument fault
rather than a result, and the second where I published before checking. The specific failure: I
observed the flat curve, remarked on it twice as "the curve is flat — the smallest budget already
saturates", and used it as *evidence* rather than asking what measurement would produce exactly
that shape. The verification I did run (`FINDING37` §2a) confirmed the *budget* was wired through
but never checked that the *evaluation* was — I verified the input and not the output.
