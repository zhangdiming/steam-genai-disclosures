import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from rule_coder import code_disclosure

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
SNAPSHOT = datetime(2026, 10, 24, tzinfo=timezone.utc)
MULTI_CATS = {1, 9, 20, 27, 36, 37, 38, 39, 47, 48, 49}


def load_jsonl(p):
    out = []
    with open(p, encoding="utf-8") as f:
        for l in f:
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    return out


def norm(t):
    return re.sub(r"\s+", " ", t.strip().lower())


def apply_runtime_pass(r, rp):
    if rp:
        r = {**r, **rp}
        if not (r["live_system"] or r["live_player"]):
            r.update({k: 0 for k in r if k.startswith("gr_")})
    return r


def llm_codes():
    A = ROOT / "annotation"
    out = {}
    m = pd.read_csv(A / "bulk/uid_map.csv")
    uid_norm = dict(zip(m.uid, m.norm))
    rp = {r.pop("uid"): r for r in load_jsonl(A / "bulk/runtime_pass.jsonl")} if (A / "bulk/runtime_pass.jsonl").exists() else {}
    for p in sorted((A / "bulk").glob("coded_batch_[0-9][0-9].jsonl")):
        for r in load_jsonl(p):
            u = r.pop("uid")
            if u in uid_norm:
                out[uid_norm[u]] = (apply_runtime_pass(r, rp.get(u)), "bulk")
    for split in ("test", "dev"):
        gp, tp = A / f"gold/gold_{split}.jsonl", A / f"gold/disclosures_{split}.jsonl"
        if not gp.exists():
            continue
        txt = {r["gid"]: r["text"] for r in load_jsonl(tp)}
        for r in load_jsonl(gp):
            g = r.pop("gid")
            if all(v is not None for v in r.values()):
                out[norm(txt[g])] = (r, f"gold_{split}")
    return out


def main():
    llm = llm_codes()
    tagname = {t["tagid"]: t["name"] for t in json.load(open(RAW / "tag_names.json", encoding="utf-8"))}
    items = {}
    for it in load_jsonl(RAW / "store_items.jsonl"):
        if not it.get("missing"):
            items[it["appid"]] = it
    pages = {}
    for p in load_jsonl(RAW / "store_pages.jsonl"):
        if "error" not in p:
            pages[p["appid"]] = p

    dev_dates = defaultdict(list)
    for a, it in items.items():
        ts = it.get("release", {}).get("steam_release_date")
        for d in it.get("basic_info", {}).get("developers", []) or []:
            if ts:
                dev_dates[d["name"].strip().lower()].append(ts)
    for k in dev_dates:
        dev_dates[k].sort()

    cohort = [int(x) for x in (ROOT / "data/processed/cohort_appids.txt").read_text().split()]
    rows = []
    for a in cohort:
        it, pg = items.get(a), pages.get(a)
        if it is None:
            continue
        rel = it.get("release", {})
        ts = rel.get("steam_release_date")
        if not ts:
            continue
        bi = it.get("basic_info", {})
        devs = [d["name"].strip() for d in bi.get("developers", []) or []]
        pubs = [d["name"].strip() for d in bi.get("publishers", []) or []]
        dev0 = devs[0].lower() if devs else f"unknown_{a}"
        prior = int(np.searchsorted(dev_dates.get(dev0, []), ts)) if devs else 0
        rv = it.get("reviews", {}).get("summary_filtered", {})
        bpo = it.get("best_purchase_option", {}) or {}
        cats = it.get("categories", {}) or {}
        players = set(cats.get("supported_player_categoryids", []) or [])
        langs = it.get("supported_languages", []) or []
        r = {
            "appid": a, "name": it.get("name"),
            "release_ts": ts, "release_date": datetime.fromtimestamp(ts, timezone.utc).date().isoformat(),
            "months_since_release": (SNAPSHOT.timestamp() - ts) / (30.44 * 86400),
            "review_count": rv.get("review_count", 0), "pct_positive": rv.get("percent_positive"),
            "early_access": int(bool(rel.get("is_early_access") or it.get("is_early_access"))),
            "price_usd": int(bpo["final_price_in_cents"]) / 100 if bpo.get("final_price_in_cents") else 0.0,
            "orig_price_usd": int(bpo["original_price_in_cents"]) / 100 if bpo.get("original_price_in_cents") else
            (int(bpo["final_price_in_cents"]) / 100 if bpo.get("final_price_in_cents") else 0.0),
            "free": int(not bpo.get("final_price_in_cents") or int(bpo.get("final_price_in_cents") or 0) == 0),
            "multiplayer": int(bool(players & MULTI_CATS)),
            "mature": int(bool(it.get("content_descriptorids"))),
            "mac": int(bool(it.get("platforms", {}).get("mac"))), "linux": int(bool(it.get("platforms", {}).get("steamos_linux"))),
            "vr": int(bool(it.get("platforms", {}).get("vr_support"))),
            "has_demo": int(bool((it.get("related_items") or {}).get("demo_appid"))),
            "n_languages": sum(1 for l in langs if l.get("supported")),
            "n_full_audio": sum(1 for l in langs if l.get("full_audio")),
            "developer": devs[0] if devs else None, "publisher": pubs[0] if pubs else None,
            "self_published": int(bool(devs) and bool(pubs) and devs[0].lower() == pubs[0].lower()),
            "dev_prior_titles": prior,
            "tags": "|".join(tagname.get(t, str(t)) for t in (it.get("tagids") or [])[:20]),
            "has_page": int(pg is not None),
        }
        if pg is not None:
            r["disclosed"] = int(bool(pg.get("ai_disclosure")))
            r["disclosure"] = pg.get("ai_disclosure")
            r["about_len"] = pg.get("about_len")
            r["page_redirect"] = int(f"/app/{a}/" not in (pg.get("final_url") or f"/app/{a}/"))
            if pg.get("ai_disclosure"):
                rc = code_disclosure(pg["ai_disclosure"])
                r.update({f"r_{k}": v for k, v in rc.items()})
                r["c_n_words"] = rc["n_words"]
                hit = llm.get(norm(pg["ai_disclosure"]))
                if hit:
                    r.update({f"c_{k}": v for k, v in hit[0].items()})
                    r["code_src"] = hit[1]
        rows.append(r)
    df = pd.DataFrame(rows)
    out = ROOT / "data/processed/games.csv"
    df.to_csv(out, index=False)
    print(df.shape, "with page:", int(df.has_page.sum()), "disclosed:", int(df.get("disclosed", pd.Series()).sum()),
          "LLM-coded:", df.get("code_src", pd.Series(dtype=str)).value_counts().to_dict())


if __name__ == "__main__":
    main()
