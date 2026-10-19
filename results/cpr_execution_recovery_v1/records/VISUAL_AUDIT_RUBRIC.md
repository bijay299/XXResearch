# Rubric (verbatim from VISUAL_AUDIT_ADDENDUM_01)

You are an image reviewer. You see only the images in this package, the exact prompt for each image and the named object. You are not told how the images were made.

## Pass A — individual-image assessment (all 208 anonymous images incl. masked M0)
Per image the reviewer sees the image, the exact prompt and the named object (no method or role labels) and records:
- Object presence of the named object: `clearly_present` / `recognizable_part_present` (clearly identifiable partial objects count as presence) / `absent` / `uncertain` (ambiguous malformed shapes are uncertain) / `unjudgeable`.
- Absolute rendering: `coherent` / `coherent_localized_defects` / `substantial_distortions_interpretable` / `pervasive_degeneration` / `uncertain_unjudgeable`.
- A brief reason for every concerning case (any rendering other than `coherent`, or `uncertain`/`unjudgeable`).
Rules: painting-like style alone is not degeneration; target absence or a different coherent scene is not automatically a rendering failure; pervasive corruption cannot establish successful erasure.
Pass A is saved and hashed (`PASS_A_JUDGMENTS.json`, SHA256 committed) before pairings are revealed.

## Pass B — paired relative-rendering assessment (192 edited vs labeled M0 reference)
Each anonymous edited image is shown beside its labeled M0 reference (matched prompt and noise) with the prompt. Methods, seeds and detector outcomes stay hidden. The reviewer judges relative rendering: `better` / `similar` / `worse` / `uncertain`, and damage extent: `none` / `localized` / `pervasive` / `uncertain`. M0 is not assumed flawless; pixel identity is not required; target absence itself is not penalized. Pass B is locked and hashed (`PASS_B_JUDGMENTS.json`) before unblinding or any comparison with detectors. Both passes are preserved without overwriting.

