# Runtime disclosures that describe only a judgment

One row per runtime disclosure of the reception sample (the 112 games of `annotation/store_about/pages.csv`), from the
disclosure text of the October 2026 crawl.

- `judge_only` = 1 when the disclosure describes runtime use of a model only to judge player input or to choose among
  authored outcomes (an answer from a fixed set, a score, a story branch, a character's next action), and describes no
  text, image, or audio that the model generates for the player. The codebook counts such use as runtime generation
  (`annotation/codebook.md`, runtime evaluation of player input); this file supports a sensitivity check only.
- `quote` is the deciding phrase; `note` gives the reason.

No disclosure in this set describes a runtime feature only as planned.

The AI assistant that ran the study read every disclosure once (21 November 2026); no human checked this reading.
`code/15_numbers.py` reports the count and the no-guardrail share without these games.
