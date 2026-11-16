# Instructions for independent coders

You are one of two independent coders. Do not look at any other coder's output, at
`code/rule_coder.py`, or at review/sales data. Read `annotation/codebook.md` in full first.

## Task 1: disclosures
Input: a JSONL file; each line has an id field (`gid` for gold samples, `uid` for bulk batches) and `text`
(one disclosure). For every line, output one JSON object (one per line, same order) with the same id field
(`gid` or `uid`, copied as given) and exactly these keys:

```
{"gid": ..., "pre": 0/1, "live_system": 0/1, "live_player": 0/1, "marketing": 0/1, "marketing_only": 0/1,
 "explicit_no_live": 0/1, "non_genai": 0/1, "authority": 0-3, "ai_on_human": 0/1, "mixed_oversight": 0/1,
 "d_visual": 0/1, "d_music": 0/1, "d_sfx": 0/1, "d_voice": 0/1, "d_text": 0/1, "d_localization": 0/1,
 "d_code": 0/1, "d_other": 0/1, "scope_minor": 0/1, "scope_extensive": 0/1, "placeholder": 0/1,
 "tools_named": 0/1, "provenance_claim": 0/1, "small_team": 0/1,
 "gr_filter": 0/1, "gr_constrained": 0/1, "gr_local": 0/1, "gr_human": 0/1, "gr_report": 0/1,
 "gr_optout": 0/1, "gr_provider": 0/1, "gr_none": 0/1}
```
Guardrail codes are 0 unless `live_system` or `live_player` is 1. `gr_none` = 1 only when runtime
generation is described and no guardrail is stated. Code the text literally; when a cue is absent, code 0.
Disclosures may be in languages other than English: code their meaning. Apply the adjudication rules listed
in block B of the codebook.

## Task 2: reviews
Input: a JSONL file; each line has `rid` and `text` (a Steam review, possibly truncated to 1,500 chars).
Output per line: `{"rid": ..., "genai_mention": 0/1, "genai_critical": 0/1}`.
- `genai_mention` = 1 if the review refers to generative AI used to make or run the game (AI art, AI
  voices, AI-written text, LLM characters, "AI slop", AI translation, the AI disclosure). Classic game AI
  (enemy/NPC behaviour, bots, difficulty, pathfinding) is 0. Mentions of AI as a story theme are 0.
- `genai_critical` = 1 if `genai_mention` = 1 and the reviewer expresses a negative evaluation of the
  generative-AI use (quality, ethics, effort, authenticity, refusal to buy). Neutral or positive mentions are 0.

Write the output file exactly at the path you are given, UTF-8, one JSON object per line, no commentary.
