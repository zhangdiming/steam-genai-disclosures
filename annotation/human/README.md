# Human validation round

Two human coders independently coded the same items without seeing each other's codes. Their labels give human–human agreement and are used to check the model codes in the paper.

## Files
| File | Contents |
|---|---|
| `annotate_template.html` | the annotation page without item data; `17_make_human_tasks.py` fills it in |
| `annotate_recode_G.html` | the second-round page with the runtime and guardrail items only (version `20261130-v2-G`, new item order) |
| `items_key.csv` | item → source id, sampling stratum (including the `decoy` flag), and number of games |
| `labels/H*_*.json` | the coders' exported labels; coders appear under the aliases H1 and H2 |
| `labels/coder*_*.jsonl` | the same labels in the format of `annotation/gold/coder{A,B}_*.jsonl`, written by `18_human_validation.py` |
| `labels/disagreements_<task>.csv` | items on which the two coders disagree, with the text, written by `18_human_validation.py` |

The first-round page (`annotate.html`) and `build/tasks.json` embed review texts and are not included;
`17_make_human_tasks.py` rebuilds them from re-collected data, and `18_human_validation.py` reads `build/tasks.json`.

## Tasks
Each coder coded every item, in an order shuffled per coder.

| Task | Items | Codes |
|---|---|---|
| Practice | 12 development-sample texts | the full codebook; not counted |
| Runtime and guardrails (`G`) | all 248 runtime texts (255 games) and 40 decoys | `live_system`, `live_player`, and the 8 guardrail codes |
| Disclosures (`D`) | 200 test-sample texts (50 with runtime cues, 150 others) | all codes of blocks A–E and the authority code |
| Reviews (`R`) | 300 reviews from the review gold sample (200 with an AI term, 100 random without) | no mention / mention, not negative / mention and negative |

Each export records, for every item, the time the item was shown in the tool (`t_ms`; each interval capped at five
minutes). Coder H1 coded the review task (`R`) first, kept a record of the codes, and then entered them in the tool in
one session, so H1's `t_ms` values for that task (median about 1 s per review, against 21 s for H2) measure entry, not
reading.

Decoys are texts with runtime cues that the model coded as not runtime. They keep coders from assuming that every item in
the runtime task is runtime generation, and they check the model for missed runtime texts.

## Second round
Agreement on the runtime task was low in the first round. Both coders then attended a calibration session based only on
the codebook, in which no item of the task was discussed and no model code was shown, and recoded all 288 items of the
task with `annotate_recode_G.html` (H1 on 27 October 2026, H2 on 30 October 2026). `18_human_validation.py` uses each
coder's latest round for each task; `18_human_validation.py --versions 20261130-v1 --out-dir results/human_round1`
reproduces the first-round statistics.

## Outputs of `18_human_validation.py`
- `results/human_validation.csv`: human–human κ and agreement for each task and code; κ and stratum-weighted precision,
  recall, and F1 of the model codes and of the LLM gold against each coder and against the human consensus.
- `results/human_guardrail_shares.csv`: share of each guardrail code among the 255 runtime games, by model, by coder,
  and by human consensus; `results/human_guardrail_shares_rs.csv`: the shares with no guardrail and with an output
  safeguard among the runtime games in the reception sample.
- `results/human_validation.json`: item counts, disagreeing cells, and the number of decoys coded as runtime generation.
