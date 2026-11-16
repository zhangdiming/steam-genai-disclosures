# Generative-AI disclosures and player reception on Steam: code and data

This repository holds the code, annotation materials, game-level data, and results of the study: collection of Steam
store data and reviews, coding of the AI-disclosure texts, validation of the coding, and the statistical analysis.

## Contents
| Folder | Contents |
|---|---|
| `code/` | the scripts, run in the order given under Pipeline |
| `annotation/` | codebook, coder instructions, gold samples, bulk coding batches and codes, model and human labels; see `annotation/README.md` |
| `data/processed/` | game-level tables: `games.csv` (disclosure texts and codes), `review_ai.csv` (review aggregates), `prior_release.csv`, `developer_lists.csv`, `cohort_appids.txt`, `review_queue.txt` |
| `data/raw/` | Internet Archive snapshot targets, dates, and display checks (`wayback_*`) |
| `results/` | analysis output |

The store and review crawls (`data/raw/search_index.jsonl`, `store_items.jsonl`, `store_pages.jsonl`,
`reviews_toprated.jsonl`) are not included; steps 1–5 re-collect them. `08_analysis.py` runs on the shared tables; the
census counts in `15_numbers.py` and the count of pages behind the age check in `08_analysis.py` need the crawls.

## Requirements
Python 3.12. Install the packages with `pip install -r requirements.txt`.

## Pipeline
Run the scripts from `code/`, in this order:

| Step | Script | Output |
|---|---|---|
| 1 | `01_enumerate_games.py` | `data/raw/search_index.jsonl`: every listing in Steam's Games category |
| 2 | `build_cohort.py` | `data/processed/cohort_appids.txt`: games released 2024-01-01 to 2026-10-24 |
| 3 | `04_fetch_store_items.py` | `data/raw/store_items.jsonl`: structured store metadata (store-browse API) |
| 4 | `02_fetch_store_pages.py` | `data/raw/store_pages.jsonl`: store pages and AI-disclosure text |
| 5 | `build_review_queue.py`, `03_fetch_reviews.py` | `data/raw/reviews_toprated.jsonl`: up to 30 most helpful English reviews per game (disclosing games and a 20% control draw) |
| 6 | `06_sample_gold.py` (dev), `06b_sample_test.py` (test; checks or rewrites `test_meta.csv`), `11_sample_review_gold.py` | gold samples in `annotation/gold/` |
| 7 | `10_make_coding_batches.py` | bulk coding batches in `annotation/bulk/` |
| 8 | `07_validate_coding.py dev\|test`, `12_validate_reviews.py` | agreement, adjudicated gold, coder validation |
| 9 | `05_build_dataset.py` | `data/processed/games.csv`: game-level table with LLM (`c_*`) and rule (`r_*`) codes |
| 10 | `09_review_ai.py` | `data/processed/review_ai.csv`: game-level shares of reviews mentioning or criticizing generative AI, reviews alleging undisclosed use, and the same flags on each review's first 1,500 characters |
| 10b | `20_prior_release.py` | `data/processed/prior_release.csv`: remasters, ports, re-releases, and series entries |
| 10c | `21_wayback_timing.py --targets`, `21_wayback_timing.py`, `21_wayback_timing.py --display` | `data/raw/wayback_targets.csv`, `data/raw/wayback_timing.jsonl`: disclosure timing from Internet Archive snapshots; `data/raw/wayback_display.csv`: how archived store pages display the disclosure |
| 10d | `23_developer_lists.py` | `data/processed/developer_lists.csv`: all developers listed for each game; must run before `08_analysis.py` |
| 11 | `08_analysis.py` | `results/`: models, robustness checks, descriptive tables, `headline.json`, `short_text.csv` (estimates with the lexicon limited to the first 1,500 characters), and `supplement.csv`; console output in `run_log.txt` |
| 11b | `22_rerun_agreement.py` | `results/rerun_agreement.csv`: test-retest agreement of the bulk coder (batch 02 coded twice); must run before `15_numbers.py` |
| 12 | `17_make_human_tasks.py`, `18_human_validation.py` | human validation round: item sets and annotation page in `annotation/human/`; `results/human_validation.csv` |
| 13 | `14_figures.py`, `15_numbers.py` | figures in `manuscript/figures/`; number macros in `manuscript/numbers.tex`; tables in `manuscript/tables/` |

The disclosure codes and the reference and human labels were produced by model annotators and human coders, not by
these scripts; the scripts write the coding inputs and read the labels back. `rule_coder.py` (rule-based disclosure
coder), `review_ai.py` (review lexicon), and `steamhttp.py` (rate-limited HTTP session) are imported by the scripts above.

## Data notes
- Snapshot: listings, store metadata, store pages, and reviews were collected on 24 October 2026 (UTC). Reviews come
  from the public Steam Community review pages.
- The crawlers limit their request rate: by default 1 to 2 requests per second, set by a command-line argument; the
  collection used at most 2.5. No account data or reviewer identifiers are stored; the review crawler uses reviewer
  profile links only to remove duplicates within a game and drops them before writing.
- Store pages that do not load without signing in (mostly the age check for mature content; `page_redirect = 1` in
  `games.csv`) carry no disclosure information; `08_analysis.py` excludes them from the cohort.
- Shared data: the game-level table with disclosure texts and codes, and game-level review aggregates. Raw review texts
  are not redistributed; step 5 re-collects them.
- Email addresses: 12 email addresses in disclosure texts and in developer or publisher names (11 games) are replaced
  by `[email removed NN]`, one number per address, in every file. The stored codes and word counts were computed on
  the original texts.
- Validation reviews: `annotation/gold/review_meta.csv` gives each review's game and its line in the original crawl
  file, which a new crawl will not reproduce, and reviews can be edited or deleted. `annotation/gold/reviews_gold_ids.jsonl`
  gives the length and SHA-256 hash of each annotated text (the review's first 1,500 characters), so a re-collected review
  can be matched to its labels. Coder agreement can be recomputed from the shared labels; lexicon precision and recall
  need the texts.
