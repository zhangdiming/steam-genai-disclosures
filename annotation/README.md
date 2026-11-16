# Annotation materials and log

## Files
- `codebook.md`, `coder_instructions.md`: codebook (blocks A–E, adjudication rules) and coder instructions.
- `gold/disclosures_dev.jsonl`, `gold/disclosures_test.jsonl`: gold samples (`gid`, `text`); `gold/disclosure_meta.csv`,
  `gold/test_meta.csv`: sampling strata and population sizes used for weighting.
- `gold/coderA_*`, `gold/coderB_*`: independent annotations; `gold/adjudicated_*`: adjudication decisions (with notes);
  `gold/gold_*`: final gold labels; `gold/agreement_*.csv`, `gold/*coder_validation_*.csv`: agreement and validation.
- `gold/reviews_gold.jsonl`, `gold/review_meta.csv`, `gold/*_reviews.jsonl`, `gold/review_validation.json`: review sample.
- `bulk/batch_XX.jsonl` (`uid`, `text`), `bulk/coded_batch_XX.jsonl`: bulk coding; `bulk/uid_map.csv`: text-to-game map
  (`uid = gold` marks texts covered by the dev gold); `bulk/runtime_pass_input.jsonl`, `bulk/runtime_pass.jsonl`: second
  pass on locus and guardrail codes for texts the bulk coder marked as runtime generation.
- `human/`: human validation round (browser annotation tool, item key, collected labels); see `human/README.md`.

## Annotators
All annotators were large language models that saw only the codebook, the instructions, and the texts:
gold samples, two independent instances of Claude Opus 5.5 (a third instance adjudicated); bulk coding, Claude Sonnet 5.5;
runtime second pass, Claude Opus 5.5. All ran on 24 October 2026 as agents with default settings.
No annotator had access to outcome data or to another annotator's codes.

## Log
- Dev round: 400 texts double-coded; 37 items (42 cells) adjudicated; adjudication rules added to block B.
- Bulk round 1 (batches 00–03): batch 02 left `d_visual` at 0 for marketing visuals, unlike the dev gold and the other
  batches. The convention (domain codes apply to marketing material) was made explicit at the end of block C, and batch 02
  was re-coded; the first pass is kept as `bulk/coded_batch_02_v1.jsonl`. Apart from `d_visual`, the two passes differ in at
  most 5.3% of cells for any code (test-retest check).
- Test sample: drawn on 24 October 2026 by `code/06b_sample_test.py` from the 1,556 unique texts of bulk batches 00–03
  (fixed seed; 100 of 228 runtime-cue texts, 300 others). Bulk validation on this sample led to the runtime second pass
  (its candidate filter was checked against gold runtime positives in both samples), so the runtime locus and guardrail
  codes are validated independently by the human coding of all runtime texts (`human/`).
- Test round: 400 texts double-coded; 26 items (30 cells) adjudicated. The two annotators shared a temporary folder, and
  coder A's assembly files overwrote coder B's first-chunk files. Coder B reported seeing only coder A's assembly script, not
  any codes. Disagreement on the affected first 50 items (0.13% of cells) was comparable to that on the remaining 350 (0.25%).
  Later annotators each used a private temporary folder.
- Bulk round 2 (batches 04–11) after the census crawl was complete; batches 07–11 consist mostly of runtime-cue texts from
  games with fewer than ten reviews, which are longer and more often state a human role.
- The runtime-cue list in `code/10_make_coding_batches.py` was extended after one runtime disclosure in the dev gold
  (a guardrail statement without runtime wording) was missed.
