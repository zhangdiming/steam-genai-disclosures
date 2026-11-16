import difflib
import importlib
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
A = importlib.import_module("08_analysis")
PUB = ROOT / "data/processed/publishers.csv"
LEGAL = set("inc llc ltd limited co corp corporation company gmbh ug ag sa sl srl sas sarl bv nv pty oy ab as aps kk "
            "plc lp sro spa kft ooo sp z o".split())
GENERIC = set("the games game studio studios entertainment interactive publishing publisher digital software "
              "productions production media team labs".split())
PLACEHOLDER = re.compile(r"(none|n a|na|null|tbd|tba|unknown|self|self published|selfpublished|independent|indie|"
                         r"indie developer|no publisher|nobody|me|myself|solo|solo developer|developer|publisher|steam)?")
CLUSTER = ("dev_key", "pub_key")


def norm(s):
    s = unicodedata.normalize("NFKC", s).lower()
    toks = re.sub(r"[^\w\s]|_", " ", s).split()
    core = [t for t in toks if t not in LEGAL and t not in GENERIC]
    return " ".join(core or [t for t in toks if t not in LEGAL] or toks)


def same(a, b):
    if a == b:
        return True
    s, l = sorted((a, b), key=len)
    if len(s) >= 4 and re.search(r"(^| )" + re.escape(s) + r"( |$)", l):
        return True
    return min(len(a), len(b)) >= 5 and difflib.SequenceMatcher(None, a, b).ratio() >= 0.8


def build_publishers():
    items = []
    with open(ROOT / "data/raw/store_items.jsonl", encoding="utf-8") as f:
        for line in f:
            try:
                it = json.loads(line)
            except json.JSONDecodeError:
                continue
            if it.get("missing"):
                continue
            bi = it.get("basic_info", {}) or {}
            dv = [d["name"].strip() for d in bi.get("developers", []) or [] if d.get("name", "").strip()]
            pb = [d["name"].strip() for d in bi.get("publishers", []) or [] if d.get("name", "").strip()]
            items.append((it["appid"], dv, [p for p in pb if not PLACEHOLDER.fullmatch(norm(p))]))
    port = defaultdict(set)
    for _, dv, pb in items:
        nd = [norm(x) for x in dv]
        for p in pb:
            np_ = norm(p)
            port[np_].update(a for a in nd if not same(np_, a))
    cohort = {int(x) for x in (ROOT / "data/processed/cohort_appids.txt").read_text().split()}
    rows = []
    for appid, dv, pb in items:
        if appid not in cohort:
            continue
        nd = [norm(x) for x in dv]
        sep = [norm(p) for p in pb if not any(same(norm(p), a) for a in nd)]
        ndev = max((len(port[p]) for p in sep), default=0)
        co = int(bool(dv) and len(sep) < len(pb))
        rows.append({"appid": appid, "publishers": json.dumps(pb, ensure_ascii=False), "pub_co": co,
                     "pub_sep": int(bool(dv) and bool(sep) and not co), "pub_est": int(bool(dv) and ndev >= 2 and not co),
                     "pub_est_any": int(bool(dv) and ndev >= 2), "pub_ndev": ndev})
    out = pd.DataFrame(rows).drop_duplicates("appid", keep="last").sort_values("appid")
    out.to_csv(PUB, index=False)
    print(len(out), "games;", int(out.pub_sep.sum()), "with a separate publisher;", int(out.pub_est.sum()), "established;",
          int(out.pub_est_any.sum()), "established incl. co-published")


def main():
    if (ROOT / "data/raw/store_items.jsonl").exists():
        build_publishers()
    df = pd.read_csv(ROOT / "data/processed/games.csv", low_memory=False)
    df = df[(df.has_page == 1) & (df.release_date >= "2024-01-01") & (df.page_redirect != 1)]
    df, tag_cols = A.prep(df)
    cols = ["pub_sep", "pub_est", "pub_est_any", "pub_co", "pub_ndev"]
    df = df.merge(pd.read_csv(PUB)[["appid"] + cols], on="appid", how="left")
    df[cols] = df[cols].fillna(0).astype(int)
    df["pub_est5"] = ((df.pub_est == 1) & (df.pub_ndev >= 5)).astype(int)
    ctrl = " + ".join([c for c in A.CONTROLS if c != "self_published"] + tag_cols)
    win = df[df.release_date <= A.WINDOW_END]
    rs = win[win.review_count >= 10]
    rd = A.add_codes(rs[(rs.disclosed == 1) & rs.code_src.notna()], "c_")
    rsr = rs.merge(pd.read_csv(ROOT / "data/processed/review_ai.csv"), on="appid", how="inner")
    rsr = rsr[rsr.n_reviews_text >= 5]
    head = json.load(open(ROOT / "results/headline.json"))
    assert len(df) == head["n_cohort_with_page"] and len(rs) == head["n_reception"] and len(rsr) == head["n_text_games"]

    H = {"n_cohort": len(df), "n_est": int(df.pub_est.sum()), "n_sep": int(df.pub_sep.sum()),
         "share_est": float(df.pub_est.mean()), "share_sep": float(df.pub_sep.mean()),
         "n_est_publishers_cohort": int(df.loc[df.pub_est == 1, "pub_key"].nunique()),
         "n_est_any": int(df.pub_est_any.sum()), "n_est_co": int(((df.pub_est_any == 1) & (df.pub_est == 0)).sum()),
         "disc_est_co": float(df.loc[(df.pub_est_any == 1) & (df.pub_est == 0), "disclosed"].mean())}
    for k, v in (("est", 1), ("self", 0)):
        H[f"disc_{k}"] = float(df.loc[df.pub_est == v, "disclosed"].mean())
        sd = rd[rd.pub_est == v]
        H[f"n_rd_{k}"] = int(len(sd))
        for y in ("any_oversight", "live_any", "marketing_only"):
            H[f"{y}_{k}"] = float(sd[y].mean())
    dn = df[df.disclosed == 1].assign(norm=lambda x: x.disclosure.str.strip().str.lower().str.replace(r"\s+", " ", regex=True))
    dev2 = dn.groupby("dev_key").filter(lambda x: len(x) >= 2)
    H["n_devs_2disc"] = int(dev2.dev_key.nunique())
    H["n_devs_2disc_reuse"] = int(dev2.groupby("dev_key").norm.agg(lambda s: s.duplicated().any()).sum())
    pe = dn[dn.pub_est == 1]
    pub2 = pe.groupby("pub_key").filter(lambda x: x.dev_key.nunique() >= 2)
    H["n_pubs_2dev"] = int(pub2.pub_key.nunique())
    H["n_pubs_2dev_games"] = int(len(pub2))
    cross = pub2.groupby(["pub_key", "norm"]).dev_key.nunique()
    H["n_pubs_2dev_shared"] = int(cross[cross >= 2].reset_index().pub_key.nunique())
    H["n_games_pub_shared"] = int(pub2.set_index(["pub_key", "norm"]).index.isin(cross[cross >= 2].index).sum())
    lv = A.add_codes(df[(df.disclosed == 1) & df.code_src.notna()], "c_")
    lv = lv[lv.live_any == 1]
    H["n_runtime_est"] = int(lv.pub_est.sum())

    rows = []

    def run(var, label, cluster=CLUSTER):
        out = []
        out += A.coef_table(A.fit(f"disclosed ~ {var} + {ctrl}", df, cluster=cluster), [var], f"{label}disclosed")
        for y in ("any_oversight", "live_any", "marketing_only"):
            out += A.coef_table(A.fit(f"{y} ~ {var} + {ctrl}", rd, cluster=cluster), [var], f"{label}{y}")
        for y, base in (("pct_positive", rs), ("crit_share", rsr)):
            m = A.fit(f"{y} ~ disclosed * {var} + {ctrl}", base, cluster=cluster)
            out += A.coef_table(m, ["disclosed", f"disclosed:{var}"], f"{label}{y}")
            L = np.zeros(len(m.params))
            L[list(m.params.index).index("disclosed")] = L[list(m.params.index).index(f"disclosed:{var}")] = 1
            t = m.t_test(L)
            lo, hi = np.squeeze(t.conf_int())
            out.append({"model": f"{label}{y}", "term": "disclosed_if_publisher", "b": float(np.squeeze(t.effect)),
                        "se": float(np.squeeze(t.sd)), "p": float(np.squeeze(t.pvalue)), "lo": float(lo), "hi": float(hi),
                        "n": int(m.nobs)})
        tests = [r for r in out if r["term"] in (var, f"disclosed:{var}")]
        for r, q in zip(tests, multipletests([r["p"] for r in tests], method="holm")[1]):
            r["p_holm"] = q
        return out
    rows += run("pub_est", "")
    rows += run("pub_sep", "broad_")
    rows += run("pub_est5", "est5_")
    rows += run("pub_est", "devcl_", cluster="dev_key")
    rows += run("pub_est_any", "any_")
    multi = df[df.groupby("dev_key").appid.transform("size") >= 2]
    both = multi.groupby("dev_key").pub_est.transform(lambda s: s.nunique() > 1)
    H["n_fe_devs_both"], H["n_fe_games_both"] = int(multi.loc[both, "dev_key"].nunique()), int(both.sum())
    rows += A.coef_table(A.fit(f"disclosed ~ pub_est + {ctrl}", multi, dev_fe=True), ["pub_est"], "devfe_disclosed")
    rows += A.coef_table(A.fit("disclosed ~ pub_est", multi, dev_fe=True), ["pub_est"], "devfe_nocov_disclosed")
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/publisher.csv", index=False)
    json.dump(H, open(ROOT / "results/publisher.json", "w"), indent=1)
    print(out[["model", "term", "b", "se", "p", "p_holm", "n"]].round(4).to_string(index=False))
    print(json.dumps(H, indent=1))


if __name__ == "__main__":
    main()
