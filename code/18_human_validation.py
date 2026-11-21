import argparse
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_ai import genai_critical, genai_mention

ROOT = Path(__file__).resolve().parents[1]
H = ROOT / "annotation/human"
G = ROOT / "annotation/gold"
GR = ["gr_filter", "gr_constrained", "gr_local", "gr_human", "gr_report", "gr_optout", "gr_provider", "gr_none"]
FULL = ["pre", "live_system", "live_player", "marketing", "marketing_only", "explicit_no_live", "non_genai",
        "ai_on_human", "mixed_oversight", "d_visual", "d_music", "d_sfx", "d_voice", "d_text", "d_localization",
        "d_code", "d_other", "scope_minor", "scope_extensive", "placeholder", "tools_named", "provenance_claim",
        "small_team"] + GR
DERIVED = ("live_any", "auth_none")
CODES = {"D": FULL + ["live_any", "auth_none", "authority"], "G": ["live_system", "live_player", "live_any"] + GR,
         "R": ["genai_mention", "genai_critical"]}
ID_KEY = {"D": "gid", "G": "item", "R": "rid"}
FILE = {"D": "disclosures", "G": "runtime", "R": "reviews"}


def jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def with_live(r):
    if r is None:
        return None
    r = dict(r)
    if r.get("live_system") is not None and r.get("live_player") is not None:
        r["live_any"] = int(bool(int(r["live_system"]) or int(r["live_player"])))
    if r.get("authority") is not None:
        r["auth_none"] = int(int(r["authority"]) == 0)
    return r


def load_humans(d, versions=None):
    raw = {}
    for p in sorted(Path(d).glob("*.json")):
        obj = json.load(open(p, encoding="utf-8"))
        if obj.get("tool") != "aidisc-annotator" or (versions and obj["version"] not in versions):
            continue
        for t, recs in obj["records"].items():
            if not recs:
                continue
            got = raw.setdefault(obj["coder"], {}).setdefault(t, {}).setdefault(obj["version"], {})
            for r in recs:
                if r["id"] not in got or r.get("ts", "") > got[r["id"]].get("ts", ""):
                    got[r["id"]] = r
    out, used = {}, {}
    for c, tasks in raw.items():
        for t, by_v in tasks.items():
            v = max(by_v)
            out.setdefault(c, {})[t] = {i: with_live(r) for i, r in by_v[v].items() if r.get("done")}
            used.setdefault(c, {})[t] = v
    return out, used


def review_texts(rids):
    spec = importlib.util.spec_from_file_location("vr", ROOT / "code/12_validate_reviews.py")
    vr = importlib.util.module_from_spec(spec); spec.loader.exec_module(vr)
    if (G / "reviews_gold.jsonl").exists():
        hashes = {r["rid"]: vr.sha(r["text"]) for r in jsonl(G / "reviews_gold.jsonl")}
    else:
        hashes = {r["rid"]: r["sha256"] for r in jsonl(G / "reviews_gold_ids.jsonl")}
    return vr.match_crawl({r: hashes[r] for r in rids}, pd.read_csv(G / "review_meta.csv").set_index("rid"))


def model_codes(key, found):
    spec = importlib.util.spec_from_file_location("bd", ROOT / "code/05_build_dataset.py")
    bd = importlib.util.module_from_spec(spec); spec.loader.exec_module(bd)
    bulk = bd.bulk_codes()
    model, gold = {"D": {}, "G": {}, "R": {}}, {"D": {}, "R": {}}
    for i in key.query("task == 'D'").item:
        if i in bulk:
            model["D"][i] = with_live(bulk[i])
    gold["D"] = {r["gid"]: with_live(r) for r in jsonl(G / "gold_test.jsonl")}
    games = pd.read_csv(ROOT / "data/processed/games.csv", low_memory=False,
                        usecols=["appid"] + ["c_" + k for k in ("live_system", "live_player", *GR)]).set_index("appid")
    for r in key.query("task == 'G'").itertuples():
        row = games.loc[int(str(r.appids).split(";")[0])]
        model["G"][r.item] = with_live({k: int(row["c_" + k]) for k in ("live_system", "live_player", *GR)})
    for i in key.query("task == 'R'").item:
        if i in found:
            text, rec = found[i]
            model["R"][i] = {"genai_mention": genai_mention(text), "genai_critical": genai_critical(text, rec)}
    gold["R"] = {r["rid"]: r for r in jsonl(G / "gold_reviews.jsonl")}
    return model, gold


def weights(key):
    w = {}
    tm = pd.read_csv(G / "test_meta.csv")
    d = key.query("task == 'D'")
    pop = {"live_cue": tm.pop_live.iloc[0], "rest": tm.pop_rest.iloc[0]}
    n = d.stratum.value_counts()
    w["D"] = {r.item: pop[r.stratum] / n[r.stratum] for r in d.itertuples()}
    rm = pd.read_csv(G / "review_meta.csv")
    r_ = key.query("task == 'R'")
    pop = {"token": rm.pop_token.iloc[0], "random_no_token": rm.pop_no_token.iloc[0]}
    n = r_.stratum.value_counts()
    w["R"] = {r.item: pop[r.stratum] / n[r.stratum] for r in r_.itertuples()}
    w["G"] = {r.item: float(r.n_games) for r in key.query("task == 'G'").itertuples()}
    return w


def kappa(a, b, k, w=None):
    if len(a) < 2 or len(set(a) | set(b)) < 2:
        return np.nan
    return cohen_kappa_score(a, b, weights="quadratic" if k == "authority" else None, sample_weight=w)


def score(t, ref, pred, items, w, live_ref):
    rows = []
    for k in CODES[t]:
        idx = [i for i in items if i in ref and i in pred and ref[i].get(k) is not None and pred[i].get(k) is not None]
        if k.startswith("gr_"):
            idx = [i for i in idx if live_ref(ref[i])]
        if not idx:
            continue
        y = np.array([int(ref[i][k]) for i in idx]); p = np.array([int(pred[i][k]) for i in idx])
        ww = np.array([w.get(i, 1.0) for i in idx])
        row = {"code": k, "n": len(idx), "kappa": kappa(y, p, k), "agree": float((y == p).mean())}
        if k != "authority":
            tp, fp, fn = (ww * (y == 1) * (p == 1)).sum(), (ww * (y == 0) * (p == 1)).sum(), (ww * (y == 1) * (p == 0)).sum()
            row.update({"n_pos_ref": int(y.sum()), "prev_ref_w": (ww * y).sum() / ww.sum(), "prev_pred_w": (ww * p).sum() / ww.sum(),
                        "P_w": tp / (tp + fp) if tp + fp else np.nan, "R_w": tp / (tp + fn) if tp + fn else np.nan,
                        "F1_w": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan})
        else:
            row["acc_w"] = (ww * (y == p)).sum() / ww.sum()
        rows.append(row)
    return rows


def consensus(t, hum, coders, items, d):
    adj_p = Path(d) / f"adjudicated_{t}.jsonl" if d else None
    adj = {r["item"]: r for r in jsonl(adj_p)} if adj_p and adj_p.exists() else {}
    cons, dis = {}, []
    for i in items:
        rec = {}
        for k in CODES[t]:
            vals = [hum[c][t][i].get(k) for c in coders]
            if any(v is None for v in vals):
                rec[k] = None
            elif len(set(vals)) == 1:
                rec[k] = int(vals[0])
            else:
                rec[k] = int(adj[i][k]) if i in adj and k in adj[i] else None
                if k not in DERIVED:
                    dis.append({"item": i, "code": k, **{c: int(v) for c, v in zip(coders, vals)},
                                "adjudicated": rec[k] if rec[k] is not None else ""})
        cons[i] = rec
    return cons, dis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", default=str(H / "labels"))
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--versions", help="comma-separated tool versions to read (e.g. 20261130-v1 for the first round); "
                    "when set, only the statistics are written (no coder files, disagreement lists or adjudication)")
    a = ap.parse_args()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    key = pd.read_csv(H / "items_key.csv", dtype={"appids": str})
    versions = a.versions.split(",") if a.versions else None
    derived = versions is None
    hum, used = load_humans(a.labels_dir, versions)
    coders = sorted(hum)
    if not coders:
        sys.exit(f"no exports in {a.labels_dir}")
    multi = len(coders) > 1
    if not multi:
        print(f"only one coder ({coders[0]}): skipping human-human agreement and consensus", file=sys.stderr)
    rids = list(key.query("task == 'R'").item)
    found = review_texts(rids)
    print(f"review-task items found in the crawl: {len(found)} of {len(rids)}", file=sys.stderr)
    model, gold = model_codes(key, found)
    w = weights(key)
    live_ref = lambda r: bool(r.get("live_any"))
    texts = {t: {x["id"]: x.get("text") for x in items}
             for t, items in json.load(open(H / "build/tasks.json", encoding="utf-8"))["tasks"].items()}
    texts["R"] = {i: v[0] for i, v in found.items()}
    rows, summary = [], {"coders": coders, "versions": used, "R_crawl_unmatched": len(rids) - len(found)}
    adj_dir = a.labels_dir if derived else None
    for t in ("D", "G", "R"):
        keyt = key.query("task == @t")
        for c in coders if derived else []:
            with open(Path(a.labels_dir) / f"coder{c}_{FILE[t]}.jsonl", "w", encoding="utf-8") as fo:
                for i in keyt.item:
                    r = hum[c].get(t, {}).get(i)
                    if r:
                        fo.write(json.dumps({ID_KEY[t]: i, **{k: r[k] for k in CODES[t] if k not in DERIVED}}) + "\n")
        items = [i for i in keyt.item if all(i in hum[c].get(t, {}) for c in coders)]
        summary[f"{t}_n_all_coders"] = len(items)
        if not items:
            continue
        for c1, c2 in itertools.combinations(coders, 2):
            for r in score(t, hum[c1][t], hum[c2][t], items, {}, live_ref):
                rows.append({"task": t, "comparison": f"{c2} vs {c1}", **r})
        refs = {c: hum[c][t] for c in coders}
        if multi:
            cons, dis = consensus(t, hum, coders, items, adj_dir)
            d = pd.DataFrame(dis)
            if dis:
                d["text"] = d.item.map(texts[t])
            if t == "D":
                aoh = d[d.code == "ai_on_human"] if dis else d
                summary["D_aoh_disagreements"] = len(aoh)
                summary["D_aoh_disagreements_translation"] = int(aoh.text.str.contains(r"translat|locali", case=False).sum()) if len(aoh) else 0
            if derived:
                d.to_csv(Path(a.labels_dir) / f"disagreements_{t}.csv", index=False)
            summary[f"{t}_disagreement_cells"] = len(dis)
            refs["consensus"] = cons
        preds = {"model": model[t], **({"llm_gold": gold[t]} if t in gold else {})}
        for rn, ref in refs.items():
            for pn, pred in preds.items():
                for r in score(t, ref, pred, items, w[t], live_ref):
                    rows.append({"task": t, "comparison": f"{pn} vs {rn}", **r})
        if t in gold:
            for r in score(t, gold[t], model[t], items, w[t], live_ref):
                rows.append({"task": t, "comparison": "model vs llm_gold", **r})
        if t == "D" and multi:
            pos = {s: {i for i in items if int(src[i]["ai_on_human"])}
                   for s, src in (("model", model[t]), ("gold", gold[t]), *((c, hum[c][t]) for c in coders))}
            extra = pos["gold"] - pos["model"]
            summary["D_aoh_model_pos"], summary["D_aoh_gold_pos"] = len(pos["model"]), len(pos["gold"])
            summary["D_aoh_model_in_gold"] = len(pos["model"] & pos["gold"])
            summary["D_aoh_gold_extra"] = len(extra)
            summary["D_aoh_gold_extra_rejected"] = len(extra - set.union(*(pos[c] for c in coders)))
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "human_validation.csv", index=False)

    rt = [i for i in key.query("task == 'G' and stratum == 'runtime'").item if all(i in hum[c].get("G", {}) for c in coders)]
    dec = [i for i in key.query("task == 'G' and stratum == 'decoy'").item if all(i in hum[c].get("G", {}) for c in coders)]
    srcs = {"model": model["G"], **{c: hum[c]["G"] for c in coders}}
    if multi:
        srcs["consensus"] = consensus("G", hum, coders, rt, adj_dir)[0] if rt else {}
    share_rows = []
    for k in ["live_any"] + GR:
        row = {"code": k}
        for s, src in srcs.items():
            idx = [i for i in rt if src.get(i, {}).get(k) is not None]
            ww = np.array([w["G"][i] for i in idx]); v = np.array([int(src[i][k]) for i in idx])
            row[f"share_{s}"] = (ww * v).sum() / ww.sum() if len(idx) else np.nan
            row[f"games_{s}"] = int(ww.sum()) if len(idx) else 0
        share_rows.append(row)
    shares = pd.DataFrame(share_rows)
    shares.to_csv(out_dir / "human_guardrail_shares.csv", index=False)

    gm = pd.read_csv(ROOT / "data/processed/games.csv", low_memory=False,
                     usecols=["appid", "release_date", "review_count"]).set_index("appid")
    in_rs = (gm.release_date <= "2026-06-30") & (gm.review_count >= 10)
    w_rs = {r.item: sum(bool(in_rs.get(int(a), False)) for a in str(r.appids).split(";") if a)
            for r in key.query("task == 'G' and stratum == 'runtime'").itertuples()}
    safety = ["gr_filter", "gr_constrained", "gr_human", "gr_report", "gr_provider"]
    rs_rows = []
    for s, src in srcs.items():
        if s == "consensus":
            continue
        idx = [i for i in rt if w_rs[i] > 0 and src.get(i, {}).get("gr_none") is not None]
        ww = np.array([w_rs[i] for i in idx])
        none = np.array([int(src[i]["gr_none"]) for i in idx])
        safe = np.array([max(int(src[i][k]) for k in safety) for i in idx])
        rs_rows.append({"source": s, "games": int(ww.sum()), "share_none": (ww * none).sum() / ww.sum(),
                        "share_output_safeguard": (ww * safe).sum() / ww.sum()})
    pd.DataFrame(rs_rows).to_csv(out_dir / "human_guardrail_shares_rs.csv", index=False)
    summary["G_runtime_items"] = len(rt)
    summary["G_decoy_items"] = len(dec)
    for c in coders:
        summary[f"G_decoys_coded_runtime_{c}"] = int(sum(hum[c]["G"][i].get("live_any", 0) for i in dec))
    json.dump({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in summary.items()},
              open(out_dir / "human_validation.json", "w"), indent=1)

    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 500)
    print(json.dumps(summary, indent=1, default=str))
    show = ["task", "comparison", "code", "n", "kappa", "agree", "P_w", "R_w", "F1_w", "acc_w"]
    print(res[[c for c in show if c in res]].round(3).to_string(index=False))
    print("\nGuardrail shares among runtime games (task G):")
    print(shares.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
