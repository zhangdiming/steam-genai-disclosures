import importlib
import json
import sys
from pathlib import Path

import pandas as pd

from review_ai import genai_accusation, genai_critical, genai_disclosure_ref, genai_mention, genai_oversight_ref

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
TOKEN = importlib.import_module("11_sample_review_gold").TOKEN
VALIDATED_CHARS = 1500


def main():
    rows = []
    with open(ROOT / "data/raw/reviews_toprated.jsonl", encoding="utf-8") as f:
        for l in f:
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            short = r["text"][:VALIDATED_CHARS]
            rows.append({"appid": r["appid"], "rec": r["recommended"], "m": genai_mention(r["text"]),
                         "c": genai_critical(r["text"], r.get("recommended")), "ct": genai_critical(r["text"]),
                         "a": genai_accusation(r["text"]),
                         "dr": genai_disclosure_ref(r["text"]), "orf": genai_oversight_ref(r["text"]),
                         "long": int(len(r["text"]) > VALIDATED_CHARS), "nt": int(not TOKEN.search(r["text"])),
                         "m_s": genai_mention(short), "c_s": genai_critical(short, r.get("recommended")),
                         "ct_s": genai_critical(short)})
    df = pd.DataFrame(rows)
    g = df.groupby("appid").agg(n_reviews_text=("m", "size"), n_mention=("m", "sum"), n_crit=("c", "sum"),
                                n_crit_term=("ct", "sum"), n_accuse=("a", "sum"), n_discref=("dr", "sum"),
                                n_overref=("orf", "sum"), n_neg=("rec", lambda s: int((s == 0).sum())))
    neg = df[df.rec == 0].groupby("appid")["c"].sum().rename("n_crit_neg")
    g = g.join(neg).fillna({"n_crit_neg": 0})
    g["mention_share"] = g.n_mention / g.n_reviews_text
    g["crit_share"] = g.n_crit / g.n_reviews_text
    g["crit_share_term"] = g.n_crit_term / g.n_reviews_text
    g["crit_share_neg"] = g.n_crit_neg / g.n_neg.where(g.n_neg > 0)
    extra = df.assign(m_late=(df.m == 1) & (df.m_s == 0), c_late=(df.c == 1) & (df.c_s == 0),
                      ct_late=(df.ct == 1) & (df.ct_s == 0), m_nt=(df.m == 1) & (df.nt == 1),
                      c_nt=(df.c == 1) & (df.nt == 1))
    g = g.join(extra.groupby("appid").agg(
        n_long=("long", "sum"), n_mention_short=("m_s", "sum"), n_crit_short=("c_s", "sum"),
        n_crit_term_short=("ct_s", "sum"), n_mention_late=("m_late", "sum"), n_crit_late=("c_late", "sum"),
        n_crit_term_late=("ct_late", "sum"), n_notoken=("nt", "sum"), n_mention_notoken=("m_nt", "sum"),
        n_crit_notoken=("c_nt", "sum")))
    g["mention_share_short"] = g.n_mention_short / g.n_reviews_text
    g["crit_share_short"] = g.n_crit_short / g.n_reviews_text
    g["crit_share_term_short"] = g.n_crit_term_short / g.n_reviews_text
    g.reset_index().to_csv(ROOT / "data/processed/review_ai.csv", index=False)
    print(len(g), "games;", len(df), "reviews; mention", df.m.mean().round(4), "critical", df.c.mean().round(4))


if __name__ == "__main__":
    main()
