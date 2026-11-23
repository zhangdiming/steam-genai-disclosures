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
- `bulk/logic_corrections.csv`: bulk codes that broke a codebook rule (marketing-only use is marketing use with no AI
  content in the game), each with the coded and corrected value and the reason, decided from the text;
  `code/05_build_dataset.py` applies them and stops if any final code set breaks a rule.
- `human/`: human validation round (browser annotation tool, item key, collected labels); see `human/README.md`.
- `store_about/`: guardrails stated elsewhere on the store page (`code/25_store_about.py`): `pages.csv` (page structure),
  `candidates.csv` (sentences of the developers' own page texts that mention AI or match a guardrail cue), `coded.csv`
  (decisions; `screen` marks the screen that found each guardrail).
- `runtime_scope/`: which of the same 112 runtime disclosures describe a model that only judges player input or
  chooses among authored outcomes (`coded.csv`; sensitivity check for the runtime denominator); see its README.

## Annotators
All annotators were large language models that saw only the codebook, the instructions, and the texts:
gold samples, two independent instances of Claude Opus 5.5; bulk coding, Claude Sonnet 5.5; runtime second pass,
Claude Opus 5.5. All ran on 24 October 2026 as agents with default settings. No annotator had access to outcome data
or to another annotator's codes.

Disagreements in the gold samples (dev, test, reviews) were adjudicated by the AI assistant that ran the study, also
Claude Opus 5.5, which drafted the codebook and knew the hypotheses; it was not a separate, isolated run. For each
disagreement it saw the text and both annotators' codes, and no outcome data. Before the dev adjudication, it had run
a test of the analysis code on partial data that printed preliminary estimates based on the rule-based coder.

## Log
- Dev round: 400 texts double-coded; 37 items (42 cells) adjudicated; adjudication rules added to block B.
- Bulk round 1 (batches 00–03): batch 02 left `d_visual` at 0 for marketing visuals, unlike the dev gold and the other
  batches. The convention (domain codes apply to marketing material) was made explicit at the end of block C, and batch 02
  was re-coded; the first pass is kept as `bulk/coded_batch_02_v1.jsonl`. Apart from `d_visual`, the two passes differ in at
  most 5.3% of cells for any code (repeat-coding check).
- Test sample: drawn on 24 October 2026 by `code/06b_sample_test.py` from the 1,556 unique texts of bulk batches 00–03
  (fixed seed; 100 of 228 runtime-cue texts, 300 others). Bulk validation on this sample led to the runtime second pass
  (its candidate filter was checked against gold runtime positives in both samples), so the runtime locus and guardrail
  codes are validated independently by the human coding of all runtime texts (`human/`).
- Rule check (17 November 2026, after review): five bulk-coded texts broke the marketing-only rule. Four describe AI
  only in marketing or community material and were set to marketing-only; one names in-game localization and was
  given the development-time code instead. Corrections in `bulk/logic_corrections.csv`.
- Test round: 400 texts double-coded; 26 items (30 cells) adjudicated. The two annotators shared a temporary folder, and
  coder A's assembly files overwrote coder B's first-chunk files. Coder B reported seeing only coder A's assembly script, not
  any codes. The assembly script only converts a file of codes to JSONL and contains no codes. Coder B continued in a
  private folder; its first 50 rows equal the first-chunk file there, which uses a format of its own. Disagreement on the
  affected first 50 items (0.13% of cells) was comparable to that on the remaining 350 (0.25%); without them, the median
  kappa is 1.00 (minimum 0.67 instead of 0.77). Later annotators each used a private temporary folder.
- Bulk round 2 (batches 04–11) after the census crawl was complete; batches 07–11 consist mostly of runtime-cue texts from
  games with fewer than ten reviews, which are longer and more often state a human role.
- The runtime-cue list in `code/10_make_coding_batches.py` was extended after one runtime disclosure in the dev gold
  (a guardrail statement without runtime wording) was missed.
- Store-page check (19 November 2026, after review): the current store pages of the 112 runtime-generation games of the
  reception sample were fetched again. The 185 sentences of the developers' own page texts (short description, "About
  This Game", mature-content description) that match a guardrail cue of the rule coder were read once, against codebook
  block E, by the AI assistant that ran the study, without a second coder; only statements about the game's generative
  system count. Decisions and quotations are in `store_about/coded.csv`.
- Store-page check, second screen (19 November 2026, after review): the first screen missed guardrails stated without a
  cue word. The screen was extended to every sentence that mentions AI, and to the developers' Early Access notes, read
  from the saved pages (617 sentences in 91 games). The same assistant read the new sentences once; general calls for
  feedback did not count as a reporting channel, and features announced for a later version did not count. Five more
  games state a guardrail (`screen` = 2), three of them coded as stating none in their disclosure.
- Store-page template check (20 November 2026, after review): `pages.csv` gained the column `ai_headings_other`, the
  number of headings outside "About This Game" and the purchase boxes (which repeat the game's name) that mention AI,
  besides the disclosure heading. It is 0 on all 112 saved pages; no decision in `coded.csv` changed.
- Runtime-scope check (21 November 2026, after review): the AI assistant that ran the study read the 112 runtime
  disclosures of the reception sample once and marked the five that describe a model only judging player input or
  choosing among authored outcomes (`runtime_scope/coded.csv`); no human checked this reading, and no code changed.
