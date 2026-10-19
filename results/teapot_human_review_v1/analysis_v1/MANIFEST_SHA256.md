# analysis_v1 — file manifest (SHA256)

Mechanical digests of every published file of this analysis, plus the analysis and amendment code under `../scripts/`.

## Coverage and exclusions — stated, not implied

Covers the **28** files listed below. Explicitly **not** covered:

* `MANIFEST_SHA256.md` — this manifest itself -- a file cannot contain its own digest
* `../EVIDENCE_INDEX.md` — the top-level evidence index, which is generated FROM the payload commit that contains this manifest and published in a child commit; it pins this manifest, not the reverse
* `amendment_public_audit.json` — the amendment's change-set privacy scan, which is run AFTER this manifest precisely so that it covers these final manifest bytes; it states its own coverage separately
* `<any later verification output>` — anything written after this manifest, including any post-index verification report, which states its own coverage separately

**No circular coverage is claimed.** Nothing in this table hashes itself, and nothing here depends on bytes created after this file.

| file | bytes | sha256 |
|---|---|---|
| `../scripts/analysis_rules.py` | 3907 | `d1150960871d4feb52c9250a737c0c3d5d0ea4b003543db4b7d418540784b8ab` |
| `../scripts/analyze_v1.py` | 27575 | `665345ab0bb02b9f554dbde4289ed2475bb858570486aa04cb789e92f1c78aac` |
| `../scripts/audit_amendment_changeset.py` | 13890 | `7a8ce482284f90fa20aea15f286136026d2e652152a1b83239921a6e70047e3a` |
| `../scripts/audit_analysis_public.py` | 10107 | `9cfbecc653598e6bae75b552b7674e6727f7718ce55188cfe3fe844630ef5d06` |
| `../scripts/finalize_analysis.py` | 7809 | `24021f81b2d746f82516f81d174a334f2245b06404edc8bf1fce7df62f58883b` |
| `../scripts/gen_analysis_manifest.py` | 3701 | `7e0b12b9831d8120503cebd2664cda6c814fa9ea9bfa96eb75c6c35217eaf13c` |
| `../scripts/gen_evidence_index.py` | 17167 | `7138a3b593cd9530281cddccbf3048ec1db4aa27b485768388d7833de797e1a6` |
| `../scripts/preservation_check.py` | 11757 | `9f0b1d4cceec9c1deb862490c7ed7306dba2b1c97bbc4c1d8c4ce5c4f00b4868` |
| `../scripts/test_analysis_synthetic.py` | 6736 | `ebf20296ab1387cfac77d50596495c6cf2942a7959d5bc7b3f585e671e01c3bf` |
| `../scripts/verify_analysis.py` | 12139 | `7a6f6b98108f1cfa0c25db070aa75d0478d47a6a2323ac98308fda566fa37286` |
| `../scripts/verify_mappings_for_analysis.py` | 7512 | `c47c6f705f09e6831918ee927da34e4291262a2f5f3eaa2cc850a1986948a059` |
| `AMENDMENT_01_CONCLUSIONS_AND_PROVENANCE.md` | 18806 | `cfaaa4a6893f9e9cae34e5e8284230810b301da494a12455716a12820335a9a8` |
| `ANALYSIS_CONTRACT.md` | 9502 | `a50a003d500cf5936742c221ed99b685b70f69172b2b381a53fbea04c801918a` |
| `FINAL_REPORT.md` | 23838 | `cd54c113c62da5ffa228d0dc68e964c055259ad6b4bf0d77e38cfe59de1e92d9` |
| `mapping_verification.json` | 3747 | `5518ee4e834d934aea560b46741dfacd11808c9d5637d5fb014d06958e2d72ad` |
| `preservation_check.json` | 2396 | `d88af8c4a98e06f44e31def739e847fae0bceca6c3a9b55fc76bddd90353b885` |
| `provenance.json` | 10554 | `8d1dc9c36c71d79c5ab08b31f3b0a6fe542a8bf57911631f81b3ff2d571a9c95` |
| `public_audit.json` | 3699 | `1f67622ce93dd1f1fed7488b8a5a555d7d1f7012d0c70695acf1cd075ca824dc` |
| `summary.json` | 11959 | `453fd1ceb01c94c68c645d0a3bd418d4b8656e6e7ecf2ca96b8c5cfd76b9de85` |
| `synthetic_checks.json` | 3208 | `4a01aea82d026dfb56c69e728333c99b1f5affd658b706a874dad463e1e7d267` |
| `tables/descriptive_strata.csv` | 392 | `a6e28aace25bbf6b49340d41917bdd551818c0954a79dba13c9820800e88978d` |
| `tables/paired_case_ledger.csv` | 2306 | `dd93bbc138b452ad5794e9f798df0e877c0fdf9f64f2c073b6147486ade78808` |
| `tables/presence_table_m0_by_edited.csv` | 139 | `d1c4db88201f2d5e3dc2de981f12f06a551b82b9a137fe349d7c90298cccd20b` |
| `tables/target_absence_rendering.csv` | 185 | `b82efb6803290e25f8b76f90e270a9bb274f1c478bc6d55e8667654d29f33e7a` |
| `tables/taskA_responses.csv` | 3755 | `c9be8806c1801a98bcb3fa95eebf803ef2276e413878f8427c2560936bdf800e` |
| `tables/taskB_distributions.csv` | 95 | `3d39c753feab0e6bfc2405ea62e2f5588995591b35b36c66eefd43802f4b7ccb` |
| `tables/taskB_responses.csv` | 923 | `c0d8e68ad0ae9877d0071226730459bb646956e87e96e3210da83b74e7d0718a` |
| `verification.json` | 7241 | `c5eb720e9185b21939eb8a22c1a525e873cda6d21a8c9c6a64691e60073f6f1d` |

28 files. Generated 2026-10-18T21:02:10.488638-04:00.
