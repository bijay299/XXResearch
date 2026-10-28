# U36 — the benign-recovery claim, checked against the images

**2026-10-28.** U34 reported that fine-tuning on non-target concepts recovers the erased concept
with no adversary. That claim rested entirely on `H_after` — the **same** closed-set CLIP metric
that U29/U30 caught scoring a visible church as `van_gogh=0.72` and a window-grid texture as
`church=0.546`. U34's own limitations recorded that the nine target panels had not been inspected.

This audit regenerates the target panel at **four independent eval seeds** from the saved
post-fine-tune checkpoints, with per-image top-3 CLIP labels, and saves the images.

## 1. The recovery replicates across eval seeds

| arm (benign fine-tuned) | target accuracy at eval seeds 0, 1, 2, 3 | mean |
|---|---|---:|
| `esd_church` | 1.0000, 0.8889, 1.0000, 0.7778 | **0.917** |
| `imma_church_frp_a1e-7_s1500` | 0.6667, 0.7778, 0.4444, 0.5556 | **0.611** |
| `uce_church` | 0.3333, 0.4444, 0.4444, 0.5556 | **0.444** |

Not a single-panel artifact. The numbers hold.

## 2. ESD — the recovery is real, and the images prove it

`outputs/audit_U34_esd_church_BENIGN_sd0_s200/sheet_teacher_target.png` shows **unambiguous
churches**: steeples, Gothic arched windows, crosses, bell towers, chapel and classical temple
facades, scored `church=1.00` repeatedly. These are not textures and not generic buildings.

**ESD's erasure is genuinely undone by fine-tuning on golf balls, french horns, cassette players
and gas pumps.** No adversary, no church prompt, no church image. This is the strongest verified
claim in the programme and it survives the image check that killed the damage-then-distil result.

## 3. UCE — the number overstates the recovery, and this cuts against us

UCE's images after benign fine-tuning are **coherent architecture**: houses, brick apartment
blocks, modern civic buildings, offices. A minority are church-like. Many generic buildings carry
labels such as `church=0.85` or `church=0.93`.

Before fine-tuning, UCE emitted abstract texture mush (U30). After, it emits coherent buildings. So
part of the measured rise from 0.000 to ~0.44 is **the return of general architectural capability
being scored as "church"**, not the specific concept returning.

**This is the U29/U30 metric defect operating in the opposite direction.** There, low scores hid a
visible concept. Here, rising scores partly reflect restored general capability. A 12-way closed set
with no reject option cannot separate "drew a church" from "drew a building".

**Consequence:** U34's headline is **narrowed**. "Benign fine-tuning recovers the erased concept" is
verified for **ESD**. For **UCE it is overstated** — what is verified is that coherent architectural
generation returns and the classifier scores it as church. The honest UCE claim is weaker and needs
a measurement the current metric cannot provide.

## 4. What this does to the project's position

It **strengthens** the central argument rather than weakening it. The programme's thesis is that the
field's instrument cannot distinguish what it claims to measure. U36 shows the defect runs **both
ways** — it hides concepts that are present and credits concepts that are absent — and we found the
second direction in our own headline result, before publishing it.

It also means the metric critique is no longer only about other people's numbers. It is about ours.

## 5. Limits

- One seed per arm audited (the highest-recovery seed of each), four eval seeds each.
- Church only, SD-1.4, 200-step benign fine-tune on 32 base-generated images.
- The base-model-generated downstream set remains a confound (U34 §limits) and is unaddressed here.
- "Church-like" versus "generic building" is **assessment by inspection of the contact sheets**, not
  a quantitative measure. The ESD case is unambiguous enough to read directly; the UCE case is a
  judgement and should be treated as provisional until measured properly.
