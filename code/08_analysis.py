import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
N_TAGS = 30
WINDOW_END = "2026-06-30"
CONTROLS = ["log_price", "free", "early_access", "log_lang", "mature", "multiplayer", "mac", "has_demo",
            "self_published", "log_dev_prior", "log_about", "C(release_month)"]
DOM = ["d_visual", "d_voice", "d_music", "d_sfx", "d_text", "d_localization", "d_code"]
EXTRA = ["ai_on_human", "scope_minor", "scope_extensive", "placeholder", "tools_named", "small_team",
         "explicit_no_live"]
AUTH = {0: "none", 1: "review", 2: "edit", 3: "human_final"}
SAFETY = ["c_gr_filter", "c_gr_constrained", "c_gr_human", "c_gr_report", "c_gr_provider"]
REREL = (r"remaster|remake|definitive|enhanced|anniversary|\bHD\b|redux|director.?s cut|complete edition|collection|"
         r"reloaded|classic")
TRANSL = r"translat|locali[sz]"
sys.path.insert(0, str(Path(__file__).resolve().parent))
LIVE_CUE = importlib.import_module("10_make_coding_batches").LIVE_CUE
C_RUNTIME, C_MKT = ("C(locus, Treatment('dev_time'))[T." + k + "]" for k in ("runtime", "marketing_only"))
FAMILIES = {"H2": lambda t: t.startswith("C(locus"), "H3": lambda t: t.startswith("C(auth") or t == "ai_on_human",
            "H4": lambda t: t in DOM}


def prep(df):
    df = df.copy()
    df["release_month"] = df["release_date"].str[:7]
    df["year"] = df["release_date"].str[:4]
    df["log_price"] = np.log1p(df["orig_price_usd"].fillna(0))
    df["log_lang"] = np.log1p(df["n_languages"].fillna(1))
    df["log_dev_prior"] = np.log1p(df["dev_prior_titles"].fillna(0))
    df["log_about"] = np.log1p(df["about_len"].fillna(0))
    df["log_reviews"] = np.log1p(df["review_count"].fillna(0))
    df["log_words"] = np.log1p(df["c_n_words"].fillna(0))
    df["dev_key"] = df["developer"].fillna("unknown_" + df["appid"].astype(str)).str.lower().str.strip()
    df["pub_key"] = df["publisher"].fillna(df["dev_key"]).str.lower().str.strip()
    df["rerelease"] = (df["name"].fillna("").str.contains(REREL, case=False)
                       | df["tags"].fillna("").str.contains(r"(^|\|)(Remake|Classic)(\||$)")).astype(int)
    tags = df["tags"].fillna("").str.split("|")
    top = pd.Series([t for ts in tags for t in ts if t]).value_counts().index[:N_TAGS]
    tag_cols = []
    for t in top:
        c = "tag_" + "".join(ch if ch.isalnum() else "_" for ch in t)
        df[c] = tags.apply(lambda ts, t=t: int(t in ts))
        tag_cols.append(c)
    return df, tag_cols


def add_codes(d, pre):
    d = d.copy()
    for k in DOM + EXTRA + ["live_system", "live_player", "marketing_only", "authority"]:
        d[k] = d[pre + k]
    d["live_any"] = ((d["live_system"] == 1) | (d["live_player"] == 1)).astype(int)
    d["locus"] = pd.Categorical(np.select([d["live_any"] == 1, d["marketing_only"] == 1],
                                          ["runtime", "marketing_only"], "dev_time"),
                                categories=["dev_time", "runtime", "marketing_only"])
    d["auth"] = pd.Categorical(d["authority"].map(AUTH), categories=list(AUTH.values()))
    d["any_oversight"] = (d["authority"] > 0).astype(int)
    return d


def groups(data, cluster):
    if isinstance(cluster, tuple):
        return np.column_stack([pd.factorize(data[c])[0] for c in cluster])
    return pd.factorize(data[cluster])[0]


def fit(formula, data, cluster="dev_key", dev_fe=False, flogit=False, weights=None):
    if weights:
        return smf.wls(formula, data=data, weights=data[weights]).fit(
            cov_type="cluster", cov_kwds={"groups": groups(data, cluster)})
    if flogit:
        data = data.assign(**{formula.split("~")[0].strip() + "_f": data[formula.split("~")[0].strip()] / 100})
        f = formula.split("~")[0].strip() + "_f ~" + formula.split("~", 1)[1]
        return smf.glm(f, data=data, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(data[cluster])[0]})
    if dev_fe:
        y, X = dmatrices(formula, data, return_type="dataframe")
        g = data.loc[y.index, cluster]
        X = X.drop(columns="Intercept")
        yd, Xd = y - y.groupby(g).transform("mean"), X - X.groupby(g).transform("mean")
        Xd = Xd.loc[:, Xd.abs().sum() > 1e-9]
        keep, basis = [], np.empty((len(Xd), 0))
        for c in Xd.columns:
            cand = np.column_stack([basis, Xd[c].values])
            if np.linalg.matrix_rank(cand, tol=1e-8 * max(1.0, np.abs(cand).max())) > basis.shape[1]:
                keep.append(c); basis = cand
        return sm.OLS(yd, Xd[keep]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(g)[0]})
    if isinstance(cluster, tuple):
        y, _ = dmatrices(formula, data, return_type="dataframe")
        data = data.loc[y.index]
    return smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": groups(data, cluster)})


def mundlak(formula, data, cluster="dev_key"):
    y, X = dmatrices(formula, data, return_type="dataframe")
    g = data.loc[y.index, cluster]
    X = X.drop(columns="Intercept")
    Z = sm.add_constant(pd.concat([X, X.groupby(g).transform("mean").add_prefix("mean_")], axis=1))
    return sm.OLS(y, Z).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(g)[0]})


def coef_table(m, terms, label):
    rows = []
    for t in terms:
        if t in m.params.index:
            ci = m.conf_int().loc[t]
            rows.append({"model": label, "term": t, "b": m.params[t], "se": m.bse[t], "p": m.pvalues[t],
                         "lo": ci[0], "hi": ci[1], "n": int(m.nobs), "r2": getattr(m, "rsquared", np.nan)})
    return rows


def within_formula(y, ctrl):
    return (f"{y} ~ C(locus, Treatment('dev_time')) + C(auth, Treatment('none')) + "
            f"{' + '.join(DOM + EXTRA + ['log_words'])} + {ctrl}")


def ingame_contrast(m, s):
    names = list(m.params.index)
    w = float((s.locus == "runtime").sum() / (s.locus != "marketing_only").sum())
    r = np.zeros(len(names))
    r[names.index("C(locus, Treatment('dev_time'))[T.marketing_only]")] = 1
    r[names.index("C(locus, Treatment('dev_time'))[T.runtime]")] = -w
    t = m.t_test(r)
    lo, hi = np.asarray(t.conf_int()).ravel()
    return {"b": float(np.squeeze(t.effect)), "lo": float(lo), "hi": float(hi), "p": float(np.squeeze(t.pvalue)),
            "w_runtime": w, "n_ref": int((s.locus == "dev_time").sum()),
            "n_ref_no_locus": int(((s.locus == "dev_time") & (s.c_pre == 0)).sum())}


def within_terms(m):
    return [t for t in m.params.index if t.startswith("C(locus") or t.startswith("C(auth")] + DOM + EXTRA + ["log_words"]


def holm(out, models):
    out["p_holm"] = np.nan
    for model in models:
        for sel in FAMILIES.values():
            mask = (out.model == model) & out.term.apply(sel)
            if mask.sum():
                out.loc[mask, "p_holm"] = multipletests(out.loc[mask, "p"], method="holm")[1]
    return out


def holm_by_term(m):
    out = {}
    for sel in FAMILIES.values():
        terms = [t for t in m.params.index if sel(t)]
        out.update(zip(terms, multipletests(m.pvalues[terms], method="holm")[1]))
    return out


def bulk_codes(bd):
    um = pd.read_csv(ROOT / "annotation/bulk/uid_map.csv")
    uid_norm = dict(zip(um.uid, um.norm))
    return {uid_norm[u]: r for u, r in bd.bulk_codes().items() if u in uid_norm}


def main():
    df = pd.read_csv(ROOT / "data/processed/games.csv", low_memory=False)
    head = {"n_pages_all": int(df["has_page"].sum())}
    df = df[(df["has_page"] == 1) & (df["release_date"] >= "2024-01-01")]
    head["n_excluded_dates"] = head["n_pages_all"] - int(len(df))
    unread = df["page_redirect"] == 1
    head["n_listed_window"] = int(len(df))
    head["n_unreadable"] = int(unread.sum())
    head["n_unreadable_mature"] = int((unread & (df["mature"] == 1)).sum())
    raw = ROOT / "data/raw/store_pages.jsonl"
    if raw.exists():
        ids, window = set(df.loc[unread, "appid"]), set(df.appid)
        with open(raw, encoding="utf-8") as f:
            pages = {r["appid"]: r for r in map(json.loads, f) if r["appid"] in window}
        urls = [pages[a].get("final_url") or "" for a in ids]
        head["n_unreadable_agecheck"] = sum("agecheck" in u for u in urls)
        app = df.appid.map(lambda a: bool(pages[a].get("is_app_page")))
        head["n_kept_not_app_page"] = int((~unread & ~app).sum())
        head["n_dropped_app_page"] = int((unread & app).sum())
    df = df[~unread]
    df, tag_cols = prep(df)
    ctrl = " + ".join(CONTROLS + tag_cols)
    pr_p = ROOT / "data/processed/prior_release.csv"
    if pr_p.exists():
        df = df.merge(pd.read_csv(pr_p)[["appid", "prior_release", "series"]], on="appid", how="left")
        df[["prior_release", "series"]] = df[["prior_release", "series"]].fillna(0).astype(int)
        df["prior_work"] = (df.prior_release | df.series).astype(int)
    scored = (df.review_count >= 10) & df.pct_positive.notna()
    own = df.pct_positive.where(scored, 0)
    cnt = scored.groupby(df.dev_key).transform("sum") - scored.astype(int)
    df["dev_track"] = (own.groupby(df.dev_key).transform("sum") - own) / cnt.where(cnt > 0)
    df["dev_track_na"] = df.dev_track.isna().astype(int)
    df["dev_track"] = df.dev_track.fillna(df.dev_track.mean())

    head["n_cohort_with_page"] = int(len(df))
    head["n_disclosed"] = int(df["disclosed"].sum())
    head["share_disclosed"] = float(df["disclosed"].mean())
    df.groupby("release_month")["disclosed"].agg(["mean", "size"]).reset_index().to_csv(
        RES / "prevalence_by_month.csv", index=False)
    for y in ("2024", "2025", "2026"):
        head[f"share_disclosed_{y}"] = float(df.loc[df.year == y, "disclosed"].mean())
    for lo, hi, name in ((0, 9, "lt10"), (10, 10**9, "ge10")):
        head[f"share_disclosed_{name}"] = float(df.loc[df.review_count.between(lo, hi), "disclosed"].mean())
    dn = df[df.disclosed == 1]
    tcount = dn["disclosure"].str.strip().str.lower().str.replace(r"\s+", " ", regex=True).value_counts()
    head["n_unique_disclosures"] = int(len(tcount))
    head["share_disclosures_shared_text"] = float((dn["disclosure"].str.strip().str.lower().str.replace(
        r"\s+", " ", regex=True).map(tcount) > 1).mean())
    head["median_words"] = float(dn["c_n_words"].median())
    dn = dn.assign(norm=dn["disclosure"].str.strip().str.lower().str.replace(r"\s+", " ", regex=True))
    shared = dn[dn.norm.map(tcount) > 1]
    one_dev = shared.groupby("norm").dev_key.transform("nunique") == 1
    head["share_shared_same_dev"] = float(one_dev.mean())

    win = df[df.release_date <= WINDOW_END].copy()
    rs = win[win.review_count >= 10].copy()
    head["n_reception"] = int(len(rs))
    head["n_reception_disclosed"] = int(rs.disclosed.sum())
    head["n_reception_devs"] = int(rs.dev_key.nunique())
    rd_all = rs[rs.disclosed == 1]
    head["n_reception_disclosed_coded"] = int(rd_all.code_src.notna().sum())
    head["code_src_reception"] = rd_all.code_src.value_counts().to_dict()

    d = add_codes(rd_all[rd_all.code_src.notna()], "c_")
    codes = [c[2:] for c in d.columns if c.startswith("c_") and c not in ("c_n_words",)]
    dist = []
    for c in codes:
        if c == "authority":
            for lvl in range(4):
                dist.append({"code": f"authority={lvl}", "share": float((d["c_authority"] == lvl).mean())})
        else:
            dist.append({"code": c, "share": float(d["c_" + c].mean())})
    pd.DataFrame(dist).to_csv(RES / "code_distribution.csv", index=False)
    head["share_visual_auth0"] = float(((d.d_visual == 1) & (d.authority == 0)).mean())
    va = (d.d_visual == 1) & (d.authority == 0)
    head["n_visual_auth0"] = int(va.sum())
    head["n_visual_auth0_mkt"] = int((va & (d.locus == "marketing_only")).sum())
    head["n_visual_auth0_aoh"] = int((va & (d.ai_on_human == 1)).sum())
    head["share_visual_auth0_no_mkt"] = float((va & (d.locus != "marketing_only")).mean())
    head["share_visual_auth0_no_aoh"] = float((va & (d.ai_on_human == 0)).mean())
    head["n_ai_on_human"] = int(d.ai_on_human.sum())
    head["n_auth0_aoh"] = int(((d.authority == 0) & (d.ai_on_human == 1)).sum())
    head["share_devvisual_auth0"] = float(((d.locus == "dev_time") & (d.d_visual == 1) & (d.authority == 0)).mean())
    head["n_live_reception"] = int(d.live_any.sum())
    d.groupby("year").agg(
        runtime=("live_any", "mean"), marketing_only=("marketing_only", "mean"),
        auth0=("authority", lambda s: (s == 0).mean()), auth1=("authority", lambda s: (s == 1).mean()),
        auth2=("authority", lambda s: (s == 2).mean()), auth3=("authority", lambda s: (s == 3).mean()),
        explicit_no_live=("explicit_no_live", "mean"), n=("appid", "size")).to_csv(RES / "codes_by_year.csv")
    pd.crosstab(d.locus, d.auth, normalize="index").to_csv(RES / "locus_by_authority.csv")
    d.locus.value_counts().rename("n").to_csv(RES / "locus_counts.csv")

    lv = add_codes(df[(df.disclosed == 1) & df.code_src.notna()], "c_")
    lv = lv[lv.live_any == 1]
    gr = [c for c in df.columns if c.startswith("c_gr_")]
    head["n_live_census"] = int(len(lv))
    head["n_live_census_unique"] = int(lv["disclosure"].str.lower().str.strip().nunique())
    head["n_live_system"] = int((lv.live_system == 1).sum())
    head["n_live_player"] = int((lv.live_player == 1).sum())
    g = lv[gr].mean().rename("share").to_frame()
    g["n"] = lv[gr].sum()
    g["share_player"] = lv.loc[lv.live_player == 1, gr].mean()
    g.to_csv(RES / "guardrails_live.csv")
    head["live_any_guardrail"] = float(lv[[c for c in gr if c != "c_gr_none"]].max(axis=1).mean())
    head["live_any_safety"] = float(lv[SAFETY].max(axis=1).mean())
    lv_in = (lv.release_date <= WINDOW_END) & (lv.review_count >= 10)
    head["n_live_in_rs"], head["n_live_out_rs"] = int(lv_in.sum()), int((~lv_in).sum())
    g["share_rs"] = lv.loc[lv_in, gr].mean()
    g.to_csv(RES / "guardrails_live.csv")
    for nm, sel in (("in", lv_in), ("out", ~lv_in)):
        head[f"live_{nm}_gr_none"] = float(lv.loc[sel, "c_gr_none"].mean())
        head[f"live_{nm}_any_safety"] = float(lv.loc[sel, SAFETY].max(axis=1).mean())
    post = lv.release_date >= "2026-01-01"
    for nm, sel in (("pre26", ~post), ("post26", post), ("in_pre26", lv_in & ~post), ("in_post26", lv_in & post),
                    ("out_pre26", ~lv_in & ~post), ("out_post26", ~lv_in & post)):
        head[f"n_live_{nm}"] = int(sel.sum())
        head[f"live_{nm}_gr_none"] = float(lv.loc[sel, "c_gr_none"].mean())
        head[f"live_{nm}_any_safety"] = float(lv.loc[sel, SAFETY].max(axis=1).mean())
    ut = lv[lv_in].assign(norm=lambda x: x.disclosure.str.strip().str.lower()).drop_duplicates("norm")
    cue = ut.disclosure.str.contains(LIVE_CUE)
    head["cue_recall_n"] = int(len(ut))
    head["cue_recall_none"] = float(cue[ut.c_gr_none == 1].mean())
    head["cue_recall_guard"] = float(cue[ut.c_gr_none == 0].mean())

    rows = []
    m1 = fit(f"pct_positive ~ disclosed + {ctrl}", rs)
    rows += coef_table(m1, ["disclosed"], "H1_pct_positive")
    m1b = fit(f"log_reviews ~ disclosed + {ctrl}", win)
    rows += coef_table(m1b, ["disclosed"], "H1_log_reviews_allwindow")
    head["raw_pct_pos_disclosed"] = float(rs.loc[rs.disclosed == 1, "pct_positive"].mean())
    head["raw_pct_pos_nondisclosed"] = float(rs.loc[rs.disclosed == 0, "pct_positive"].mean())

    rd = add_codes(rd_all[rd_all.code_src.notna()], "c_")
    f_w = within_formula("pct_positive", ctrl)
    m2 = fit(f_w, rd)
    rows += coef_table(m2, within_terms(m2), "within_pct_positive")
    head["h2a_ingame_pct_positive"] = ingame_contrast(m2, rd)
    m2b = fit(f"pct_positive ~ any_oversight + C(locus, Treatment('dev_time')) + "
              f"{' + '.join(DOM + EXTRA + ['log_words'])} + {ctrl}", rd)
    rows += coef_table(m2b, ["any_oversight"], "within_pct_positive_anyoversight")
    m2c = fit(within_formula("log_reviews", ctrl), rd)
    rows += coef_table(m2c, within_terms(m2c), "within_log_reviews")
    ov = rd.assign(n_domains=rd[DOM].sum(axis=1) + rd["c_d_other"], post26=(rd.release_date >= "2026-01-01").astype(int))
    rows += coef_table(fit("any_oversight ~ n_domains + scope_minor + scope_extensive + post26 + "
                          "C(locus, Treatment('dev_time'))", ov), ["n_domains", "scope_minor", "scope_extensive", "post26"],
                      "oversight_extent")
    head["ov_share"] = float(ov.any_oversight.mean())
    head["ov_dom1"] = float(ov.loc[ov.n_domains == 1, "any_oversight"].mean())
    head["ov_dom3"] = float(ov.loc[ov.n_domains >= 3, "any_oversight"].mean())
    head["ov_pre26"] = float(ov.loc[ov.post26 == 0, "any_oversight"].mean())
    head["ov_post26"] = float(ov.loc[ov.post26 == 1, "any_oversight"].mean())
    head["share_small_team"] = float(rd.small_team.mean())

    ra_p = ROOT / "data/processed/review_ai.csv"
    rsr = rdr = None
    if ra_p.exists():
        ra = pd.read_csv(ra_p)
        head["n_reviews_collected"] = int(ra.loc[ra.appid.isin(df.appid), "n_reviews_text"].sum())
        rsr = rs.merge(ra, on="appid", how="inner")
        rsr = rsr[rsr.n_reviews_text >= 5]
        head["n_text_games"] = int(len(rsr))
        head["n_text_games_disclosed"] = int(rsr.disclosed.sum())
        head["n_text_reviews"] = int(rsr.n_reviews_text.sum())
        for k in ("crit_share", "mention_share", "crit_share_term"):
            m3 = fit(f"{k} ~ disclosed + {ctrl}", rsr)
            rows += coef_table(m3, ["disclosed"], f"H1_{k}")
            head[f"{k}_disclosed"] = float(rsr.loc[rsr.disclosed == 1, k].mean())
            head[f"{k}_nondisclosed"] = float(rsr.loc[rsr.disclosed == 0, k].mean())
        head["share_disclosed_any_crit"] = float((rsr.loc[rsr.disclosed == 1, "n_crit"] > 0).mean())
        head["share_nondisclosed_any_crit"] = float((rsr.loc[rsr.disclosed == 0, "n_crit"] > 0).mean())
        rdr = rd.merge(ra, on="appid", how="inner")
        rdr = rdr[rdr.n_reviews_text >= 5].assign(crit_given_mention=lambda x: x.n_crit / x.n_mention.where(x.n_mention > 0))
        m4 = fit(within_formula("crit_share", ctrl), rdr)
        rows += coef_table(m4, within_terms(m4), "within_crit_share")
        head["h2a_ingame_crit_share"] = ingame_contrast(m4, rdr)
        rdm = rdr.dropna(subset=["crit_given_mention"])
        head["n_text_games_disc_mention"] = int(len(rdm))
        for k, sd in (("mention_share", rdr), ("crit_given_mention", rdm)):
            m = fit(within_formula(k, ctrl), sd)
            rows += coef_table(m, within_terms(m), f"within_{k}")
            for lv in ("dev_time", "runtime"):
                head[f"{k}_{lv}"] = float(sd.loc[sd.locus == lv, k].mean())

    out = holm(pd.DataFrame(rows), ("within_pct_positive", "within_crit_share", "within_log_reviews",
                                    "within_mention_share", "within_crit_given_mention"))
    out.to_csv(RES / "models.csv", index=False)
    pd.set_option("display.width", 220)
    print(out[["model", "term", "b", "se", "p", "p_holm", "n"]].round(4).to_string(index=False))

    key = ["disclosed", "C(locus, Treatment('dev_time'))[T.runtime]", "C(locus, Treatment('dev_time'))[T.marketing_only]",
           "C(auth, Treatment('none'))[T.review]", "C(auth, Treatment('none'))[T.edit]",
           "C(auth, Treatment('none'))[T.human_final]", "ai_on_human", "d_visual", "d_voice"]
    rob = []
    n_dev = rs.groupby("dev_key").appid.transform("size")
    variants = {
        "main": dict(),
        "min_reviews_5": dict(sub=win.review_count >= 5, base=win),
        "min_reviews_20": dict(sub=rs.review_count >= 20),
        "min_reviews_50": dict(sub=rs.review_count >= 50),
        "no_early_access": dict(sub=rs.early_access == 0, drop="early_access"),
        "excl_prolific_devs_gt5": dict(sub=n_dev <= 5),
        "year_2024": dict(sub=rs.year == "2024"),
        "year_2025": dict(sub=rs.year == "2025"),
        "year_2026h1": dict(sub=rs.year == "2026"),
        "fractional_logit": dict(flogit=True),
        "rule_codes": dict(pre="r_"),
        "excl_rereleases": dict(sub=rs.rerelease == 0),
        "track_record": dict(add="dev_track + dev_track_na"),
        "prior_work_control": dict(add="prior_release + series"),
        "excl_prior_work": dict(sub=rs.prior_work == 0),
        "multi_game_devs_pooled": dict(multi=True),
        "developer_fe": dict(dev_fe=True, multi=True),
    }
    for name, v in variants.items():
        base = v.get("base", rs)
        sub = base[v["sub"]] if "sub" in v else base
        kw = {k: v[k] for k in ("flogit", "dev_fe") if k in v}
        ctrl_v = " + ".join(c for c in CONTROLS + tag_cols if c != v.get("drop")) + (" + " + v["add"] if "add" in v else "")
        if v.get("multi"):
            sub = sub[sub.groupby("dev_key").appid.transform("size") >= 2]
        try:
            m = fit(f"pct_positive ~ disclosed + {ctrl_v}", sub, **kw)
            rob += coef_table(m, ["disclosed"], name)
        except Exception as e:
            print("robustness H1 failed:", name, e)
        sd = sub[sub.disclosed == 1]
        pre = v.get("pre", "c_")
        if pre == "c_":
            sd = sd[sd.code_src.notna()]
        try:
            sd = add_codes(sd, pre)
            if v.get("multi"):
                sd = sd[sd.groupby("dev_key").appid.transform("size") >= 2]
            if v.get("dev_fe"):
                ind = pd.DataFrame({"runtime": sd.locus.astype(str) == "runtime",
                                    "marketing_only": sd.locus.astype(str) == "marketing_only",
                                    **{a: sd.auth.astype(str) == a for a in ("review", "edit", "human_final")},
                                    "ai_on_human": sd.ai_on_human == 1, "d_visual": sd.d_visual == 1})
                nv = ind.groupby(sd.dev_key.values).nunique().gt(1).sum()
                head["fe_disc_devs"] = int(sd.dev_key.nunique())
                head["fe_disc_vary"] = {k: int(x) for k, x in nv.items()}
            m = fit(within_formula("pct_positive", ctrl_v), sd, **kw)
            rob += coef_table(m, key[1:], name)
        except Exception as e:
            print("robustness within failed:", name, e)
    if rdr is not None:
        for name, sd in (("crit_rule_codes", add_codes(rs[rs.disclosed == 1], "r_").merge(ra, on="appid")),
                         ("crit_min_reviews_20", rdr[rdr.review_count >= 20])):
            sd = sd[sd.n_reviews_text >= 5]
            m = fit(within_formula("crit_share", ctrl), sd)
            rob += coef_table(m, key[1:], name)
    mg_all = rs[rs.groupby("dev_key").appid.transform("size") >= 2]
    sw = mg_all.groupby("dev_key").disclosed.agg(["min", "max"])
    sw = sw[(sw["min"] == 0) & (sw["max"] == 1)].index
    head["fe_devs"] = int(mg_all.dev_key.nunique())
    head["fe_switch_devs"] = int(len(sw))
    head["fe_switch_games"] = int(mg_all.dev_key.isin(sw).sum())
    mgw = win[win.groupby("dev_key").appid.transform("size") >= 2]
    rob += coef_table(fit(f"log_reviews ~ disclosed + {ctrl}", mgw), ["disclosed"], "log_reviews_multi_pooled")
    rob += coef_table(fit(f"log_reviews ~ disclosed + {ctrl}", mgw, dev_fe=True), ["disclosed"], "log_reviews_developer_fe")
    first = mg_all[mg_all.disclosed == 1].groupby("dev_key").release_date.min()
    nd = mg_all[mg_all.dev_key.isin(sw) & (mg_all.disclosed == 0)]
    head["fe_nondisc_n"] = int(len(nd))
    head["fe_nondisc_before"] = float((nd.release_date < nd.dev_key.map(first)).mean())
    rob += coef_table(fit(f"pct_positive ~ disclosed + {ctrl}", rs, cluster=("dev_key", "pub_key")), ["disclosed"],
                      "cluster_dev_pub")
    rob += coef_table(fit(within_formula("pct_positive", ctrl), rd, cluster=("dev_key", "pub_key")), key[1:],
                      "cluster_dev_pub")
    rda = rd.assign(aoh_transl=lambda x: ((x.ai_on_human == 1) & x.disclosure.str.contains(TRANSL, case=False)).astype(int),
                    aoh_other=lambda x: ((x.ai_on_human == 1) & ~x.disclosure.str.contains(TRANSL, case=False)).astype(int))
    head["aoh_n"], head["aoh_transl_n"] = int(rda.ai_on_human.sum()), int(rda.aoh_transl.sum())
    head["aoh_rerelease_n"] = int(((rda.ai_on_human == 1) & (rda.rerelease == 1)).sum())
    f_aoh = within_formula("pct_positive", ctrl).replace("ai_on_human", "aoh_transl + aoh_other")
    rob += coef_table(fit(f_aoh, rda), ["aoh_transl", "aoh_other"], "aoh_split")
    def no_marketing(d):
        return d[d.c_marketing == 0].assign(locus=lambda x: x.locus.astype(str))
    rob += coef_table(fit(within_formula("pct_positive", ctrl), no_marketing(rd)), ["d_visual"], "visual_no_marketing")
    if rdr is not None:
        rob += coef_table(fit(within_formula("crit_share", ctrl), rdr), key[1:], "crit_main")
        rob += coef_table(fit(within_formula("mention_share", ctrl), rdr), key[1:], "crit_mention_share")
        rob += coef_table(fit(within_formula("crit_given_mention", ctrl), rdm), key[1:], "crit_given_mention")
        rob += coef_table(fit(within_formula("crit_given_mention", ctrl), rdm, weights="n_mention"), key[1:],
                          "crit_given_mention_weighted")
        rob += coef_table(fit(within_formula("crit_share_term", ctrl), rdr), key[1:], "crit_term")
        rob += coef_table(fit(within_formula("crit_share", ctrl), rdr, cluster=("dev_key", "pub_key")), key[1:],
                          "crit_cluster_dev_pub")
        rob += coef_table(fit(within_formula("crit_share", ctrl + " + dev_track + dev_track_na"), rdr), key[1:],
                          "crit_track_record")
        rob += coef_table(fit(within_formula("crit_share", ctrl), no_marketing(rdr)), ["d_visual"],
                          "crit_visual_no_marketing")
        head["share_disc_track"] = float(1 - rd.dev_track_na.mean())
        head["aoh_prior_work_n"] = int(((rd.ai_on_human == 1) & (rd.prior_work == 1)).sum())
        head["disc_prior_work_n"] = int(rd.prior_work.sum())
    if rsr is not None:
        mg = rsr[rsr.groupby("dev_key").appid.transform("size") >= 2]
        swt = mg.groupby("dev_key").disclosed.agg(["min", "max"])
        head["fe_crit_switch_devs"] = int(((swt["min"] == 0) & (swt["max"] == 1)).sum())
        rob += coef_table(fit(f"crit_share ~ disclosed + {ctrl}", rsr), ["disclosed"], "crit_main")
        rob += coef_table(fit(f"crit_share_term ~ disclosed + {ctrl}", rsr), ["disclosed"], "crit_term")
        for k in ("crit_share", "mention_share", "crit_share_term"):
            rob += coef_table(fit(f"{k} ~ disclosed + {ctrl}", mg), ["disclosed"], f"{k}_multi_game_devs_pooled")
            rob += coef_table(fit(f"{k} ~ disclosed + {ctrl}", mg, dev_fe=True), ["disclosed"], f"{k}_developer_fe")
        rob += coef_table(fit(f"crit_share ~ disclosed + {ctrl} + dev_track + dev_track_na", rsr), ["disclosed"],
                          "crit_track_record")
        head["accuse_games_disc"] = int((rsr.loc[rsr.disclosed == 1, "n_accuse"] > 0).sum())
        head["accuse_games_non"] = int((rsr.loc[rsr.disclosed == 0, "n_accuse"] > 0).sum())
        head["accuse_reviews"] = int(rsr.n_accuse.sum())
        head["accuse_share_of_crit_disc"] = float(rsr.loc[rsr.disclosed == 1, "n_accuse"].sum()
                                                  / rsr.loc[rsr.disclosed == 1, "n_crit"].sum())
        ov = rdr.authority > 0
        head["exp_reviews"] = int(rdr.n_mention.sum())
        head["exp_cite"] = float(rdr.n_discref.sum() / rdr.n_mention.sum())
        head["exp_cite_none"] = float(rdr.n_discref[~ov].sum() / rdr.n_mention[~ov].sum())
        head["exp_cite_oversight"] = float(rdr.n_discref[ov].sum() / rdr.n_mention[ov].sum())
        head["exp_reviews_oversight"] = int(rdr.n_mention[ov].sum())
        head["exp_overref_oversight"] = int(rdr.n_overref[ov].sum())
        na = rsr[rsr.n_accuse == 0]
        mgn = na[na.groupby("dev_key").appid.transform("size") >= 2]
        rob += coef_table(fit(f"crit_share ~ disclosed + {ctrl}", na), ["disclosed"], "crit_noaccuse")
        rob += coef_table(fit(f"crit_share ~ disclosed + {ctrl}", mgn), ["disclosed"], "crit_noaccuse_multi_pooled")
        rob += coef_table(fit(f"crit_share ~ disclosed + {ctrl}", mgn, dev_fe=True), ["disclosed"],
                          "crit_noaccuse_developer_fe")
        swn = mgn.groupby("dev_key").disclosed.agg(["min", "max"])
        head["fe_crit_noaccuse_switch_devs"] = int(((swn["min"] == 0) & (swn["max"] == 1)).sum())
        for y in ("2024", "2025", "2026"):
            rob += coef_table(fit(f"crit_share ~ disclosed + {ctrl}", rsr[rsr.year == y]), ["disclosed"], f"crit_year_{y}")
        dp = ROOT / "data/raw/wayback_display.csv"
        if dp.exists():
            dp = pd.read_csv(dp)
            head["disp_games"], head["disp_pages"] = int(dp.appid.nunique()), int(len(dp))
            head["disp_pre"], head["disp_post"] = int((dp.date < "2026").sum()), int((dp.date >= "2026").sum())
            head["disp_lead_in_all"] = bool(dp.lead_in.all())
            head["disp_no_guardrail_field"] = bool((dp.guardrail_label == 0).all() and (dp.headings == 0).all())
        wb_p = ROOT / "data/raw/wayback_timing.jsonl"
        if wb_p.exists():
            wb = pd.read_json(wb_p, lines=True)
            for grp in ("fe", "random"):
                w = wb[wb.group == grp]
                head[f"wb_{grp}_n"] = int(len(w))
                for stt in ("early", "late", "uncertain", "unknown"):
                    head[f"wb_{grp}_{stt}"] = int((w.status == stt).sum())
            late, early = set(wb.appid[wb.status == "late"]), set(wb.appid[wb.status == "early"])
            mgt = rsr[rsr.groupby("dev_key").appid.transform("size") >= 2]
            swt = mgt.groupby("dev_key").disclosed.agg(["min", "max"])
            swt = swt[(swt["min"] == 0) & (swt["max"] == 1)].index
            elig, fe_t = set(mgt.appid[mgt.dev_key.isin(swt) & (mgt.disclosed == 1)]), set(wb.appid[wb.group == "fe"])
            head["wb_fe_eligible"], head["wb_fe_extra"] = len(fe_t & elig), len(fe_t - elig)
            head["wb_fe_eligible_missing"] = len(elig - fe_t)
            for name, keep in (("nolate", ~rsr.appid.isin(late)), ("early", (rsr.disclosed == 0) | rsr.appid.isin(early))):
                sub = rsr[keep]
                sub = sub[sub.groupby("dev_key").appid.transform("size") >= 2]
                rob += coef_table(fit(f"crit_share ~ disclosed + {ctrl}", sub, dev_fe=True), ["disclosed"],
                                  f"crit_wb_{name}_developer_fe")
                s_ = sub.groupby("dev_key").disclosed.agg(["min", "max"])
                head[f"wb_{name}_switch_devs"] = int(((s_["min"] == 0) & (s_["max"] == 1)).sum())
        for lab, sd in (("all", rsr), ("disc", rsr[rsr.disclosed == 1]), ("non", rsr[rsr.disclosed == 0])):
            head[f"long_share_{lab}"] = float(sd.n_long.sum() / sd.n_reviews_text.sum())
            for k in ("mention", "crit", "crit_term"):
                head[f"late_share_{k}_{lab}"] = float(sd[f"n_{k}_late"].sum() / sd[f"n_{k}"].sum())
        head["notoken_reviews"] = int(rsr.n_notoken.sum())
        head["notoken_mention"], head["notoken_crit"] = int(rsr.n_mention_notoken.sum()), int(rsr.n_crit_notoken.sum())
        head["notoken_crit_share"] = float(rsr.n_crit_notoken.sum() / rsr.n_crit.sum())
        rds = rdr.assign(crit_given_mention_short=lambda x: x.n_crit_short / x.n_mention_short.where(x.n_mention_short > 0))
        short = []
        for y, name, sd in (("crit_share_short", "crit_short", rds), ("mention_share_short", "mention_short", rds),
                            ("crit_share_term_short", "crit_term_short", rds),
                            ("crit_given_mention_short", "crit_given_mention_short",
                             rds.dropna(subset=["crit_given_mention_short"]))):
            if y != "crit_given_mention_short":
                short += coef_table(fit(f"{y} ~ disclosed + {ctrl}", rsr), ["disclosed"], name)
            m = fit(within_formula(y, ctrl), sd)
            short += coef_table(m, within_terms(m), name)
        for k in ("crit_share_short", "mention_share_short", "crit_share_term_short"):
            short += coef_table(fit(f"{k} ~ disclosed + {ctrl}", mg, dev_fe=True), ["disclosed"], f"{k}_developer_fe")
        holm(pd.DataFrame(short), ("crit_short", "mention_short", "crit_term_short", "crit_given_mention_short")).to_csv(
            RES / "short_text.csv", index=False)
    supp = []
    head["top_tags"] = [c[4:] for c in tag_cols]
    lr = win[(win.disclosed == 1) & win.review_count.between(5, 9)]
    head["lowrev_disc_n"], head["lowrev_disc_coded"] = int(len(lr)), int(lr.code_src.notna().sum())
    head["clusters"] = {"reception": int(rs.dev_key.nunique()), "window": int(win.dev_key.nunique()),
                        "within": int(rd.dev_key.nunique()), "multi_game": int(mg_all.dev_key.nunique())}
    supp += coef_table(mundlak(f"pct_positive ~ disclosed + {ctrl}", mg_all), ["disclosed", "mean_disclosed"],
                       "mundlak_pct_positive")
    ctrl_nt = " + ".join(CONTROLS)
    supp += coef_table(fit(f"pct_positive ~ disclosed + {ctrl_nt}", rs), ["disclosed"], "notags_H1_pct_positive")
    supp += coef_table(fit(f"pct_positive ~ disclosed + {ctrl_nt}", mg_all, dev_fe=True), ["disclosed"],
                       "notags_H1_pct_positive_developer_fe")
    m_nt = fit(within_formula("pct_positive", ctrl_nt), rd)
    supp += coef_table(m_nt, within_terms(m_nt), "notags_within_pct_positive")
    head["ov_r2_within"] = float(fit(f"any_oversight ~ C(locus, Treatment('dev_time')) + "
                                     f"{' + '.join(DOM + EXTRA + ['log_words'])} + {ctrl}", rd).rsquared)
    loc = rda[rda.d_localization == 1]
    head["loc_n"], head["loc_aoh_transl_n"] = int(len(loc)), int(loc.aoh_transl.sum())
    if rsr is not None:
        head["clusters"]["text"] = int(rsr.dev_key.nunique())
        head["clusters"]["within_text"] = int(rdr.dev_key.nunique())
        supp += coef_table(mundlak(f"crit_share ~ disclosed + {ctrl}", mg), ["disclosed", "mean_disclosed"],
                           "mundlak_crit_share")
        supp += coef_table(fit(f"pct_positive ~ disclosed + {ctrl}", mg), ["disclosed"], "crit_sub_pct_positive_pooled")
        supp += coef_table(fit(f"pct_positive ~ disclosed + {ctrl}", mg, dev_fe=True), ["disclosed"],
                           "crit_sub_pct_positive_developer_fe")
        supp += coef_table(fit(f"crit_share ~ disclosed + {ctrl_nt}", rsr), ["disclosed"], "notags_H1_crit_share")
        supp += coef_table(fit(f"crit_share ~ disclosed + {ctrl_nt}", mg, dev_fe=True), ["disclosed"],
                           "notags_H1_crit_share_developer_fe")
        m_nt = fit(within_formula("crit_share", ctrl_nt), rdr)
        supp += coef_table(m_nt, within_terms(m_nt), "notags_within_crit_share")
        rde = rdr.assign(eng_pos=100 * (1 - rdr.n_neg / rdr.n_reviews_text))
        m_e = fit(within_formula("eng_pos", ctrl), rde)
        supp += coef_table(m_e, within_terms(m_e), "within_eng_pos")
        head["eng_pos_disclosed_mean"] = float(rde.eng_pos.mean())
    rb = rd[rd.code_src == "bulk"]
    m_b = fit(within_formula("pct_positive", ctrl), rb)
    supp += coef_table(m_b, within_terms(m_b), "bulkonly_within_pct_positive")
    head["bulkonly_n"] = int(len(rb))

    bd = importlib.import_module("05_build_dataset")
    bulk = bulk_codes(bd)
    rl = rd.assign(norm=rd.disclosure.map(bd.norm))
    rl["b_aoh"] = rl.norm.map(lambda t: bulk[t]["ai_on_human"] if t in bulk else np.nan)
    assert rl.loc[rl.code_src == "gold_dev", "b_aoh"].isna().all() and rl.loc[rl.code_src != "gold_dev", "b_aoh"].notna().all()
    tg = rl[rl.code_src == "gold_test"]
    head["label_test_games"] = int(len(tg))
    head["label_test_ref_pos"], head["label_test_bulk_pos"] = int(tg.ai_on_human.sum()), int(tg.b_aoh.sum())
    head["label_test_bulk_outside_ref"] = int(((tg.b_aoh == 1) & (tg.ai_on_human == 0)).sum())
    for k, sel in (("ref_only", (tg.ai_on_human == 1) & (tg.b_aoh == 0)), ("both", (tg.ai_on_human == 1) & (tg.b_aoh == 1)),
                   ("neither", (tg.ai_on_human == 0) & (tg.b_aoh == 0))):
        head[f"label_test_pos_{k}"] = float(tg.loc[sel, "pct_positive"].mean())
    G = ROOT / "annotation/gold"
    texts = {r["gid"]: bd.norm(r["text"]) for r in bd.load_jsonl(G / "disclosures_test.jsonl")}
    ca, cb = ({r["gid"]: r["ai_on_human"] for r in bd.load_jsonl(G / f"coder{c}_disclosures_test.jsonl")} for c in "AB")
    both_annot = {t: int(ca[g] and cb[g]) for g, t in texts.items()}
    head["label_test_ref_equals_both_annotators"] = int((tg.norm.map(both_annot) == tg.ai_on_human).sum())
    nod = rl[rl.code_src != "gold_dev"]
    allbulk = nod.copy()
    for c in DOM + EXTRA + ["live_system", "live_player", "marketing_only", "authority"]:
        allbulk["c_" + c] = [bulk[t][c] for t in allbulk.norm]
    versions = {"keepdev": rl.assign(ai_on_human=rl.b_aoh.fillna(rl.ai_on_human).astype(int)),
                "devzero": rl.assign(ai_on_human=rl.b_aoh.fillna(0).astype(int)),
                "nodev": nod.assign(ai_on_human=nod.b_aoh.astype(int)),
                "allbulk": add_codes(allbulk, "c_")}
    h_base = holm_by_term(m2)
    for name, x in versions.items():
        mv = fit(within_formula("pct_positive", ctrl), x)
        supp += coef_table(mv, ["ai_on_human"], f"label_{name}")
        hv = holm_by_term(mv)
        head[f"label_{name}_share"] = float(x.ai_on_human.mean())
        head[f"label_{name}_holm"] = float(hv["ai_on_human"])
        head[f"label_{name}_music_holm"] = float(hv["d_music"])
        head[f"label_{name}_holm_changes"] = [t for t in hv if (hv[t] < .05) != (h_base[t] < .05)]
        if rdr is not None:
            xr = x.merge(ra, on="appid", how="inner")
            mc = fit(within_formula("crit_share", ctrl), xr[xr.n_reviews_text >= 5])
            supp += coef_table(mc, ["ai_on_human"], f"label_{name}_crit")
            hc, hc0 = holm_by_term(mc), holm_by_term(m4)
            head[f"label_{name}_crit_holm"] = float(hc["ai_on_human"])
            head[f"label_{name}_crit_holm_changes"] = [t for t in hc if (hc[t] < .05) != (hc0[t] < .05)]
            head[f"label_{name}_crit_visual_holm"] = float(hc["d_visual"])
            supp += coef_table(mc, ["d_visual", C_RUNTIME, C_MKT], f"label_{name}_crit")

    def aoh_split(x):
        d = x.assign(aoh_transl=((x.ai_on_human == 1) & x.disclosure.str.contains(TRANSL, case=False)).astype(int),
                     aoh_other=((x.ai_on_human == 1) & ~x.disclosure.str.contains(TRANSL, case=False)).astype(int))
        ms = fit(f_aoh, d)
        return ms, int(d.aoh_transl.sum()), int(d.aoh_other.sum()), float(np.squeeze(ms.t_test("aoh_transl - aoh_other").pvalue))
    _, _, head["aoh_split_other_n"], head["aoh_split_equal_p"] = aoh_split(rd)
    ms, head["label_nodev_transl_n"], head["label_nodev_other_n"], head["label_nodev_split_equal_p"] = aoh_split(versions["nodev"])
    supp += coef_table(ms, ["aoh_transl", "aoh_other"], "label_nodev_split")

    dl_p = ROOT / "data/processed/developer_lists.csv"
    if dl_p.exists():
        dl = pd.read_csv(dl_p)
        devs = dict(zip(dl.appid, dl.developers.map(json.loads)))
        parent = {}

        def find(a):
            parent.setdefault(a, a)
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for a in df.appid:
            names = devs.get(a, [])
            for u, v in zip(names, names[1:]):
                parent[find(u)] = find(v)

        def n_dev(x):
            return x.appid.map(lambda a: len(devs.get(a, [])))

        def multi(x):
            return x[x.groupby("dev_key").appid.transform("size") >= 2]
        units = {"single": lambda x: x[n_dev(x) <= 1],
                 "last": lambda x: x.assign(dev_key=[devs[a][-1] if devs.get(a) else k for a, k in zip(x.appid, x.dev_key)]),
                 "linked": lambda x: x.assign(dev_key=[find(devs[a][0]) if devs.get(a) else k for a, k in zip(x.appid, x.dev_key)])}
        outcomes = [("pct_positive", rs)] + ([("crit_share", rsr), ("crit_share_term", rsr)] if rsr is not None else [])
        for y, base in outcomes:
            for name, unit in units.items():
                supp += coef_table(fit(f"{y} ~ disclosed + {ctrl}", multi(unit(base)), dev_fe=True), ["disclosed"],
                                   f"devunit_{name}_{y}")
        head["dev_multi_share_reception"] = float((n_dev(rs) >= 2).mean())
        head["dev_multi_share_fe"] = float((n_dev(multi(rs)) >= 2).mean())
    sp = pd.DataFrame(supp)
    sp.to_csv(RES / "supplement.csv", index=False)
    print(sp[["model", "term", "b", "se", "p", "n"]].round(4).to_string(index=False))

    rob = pd.DataFrame(rob)
    rob.to_csv(RES / "robustness.csv", index=False)
    print(rob[["model", "term", "b", "se", "p", "n"]].round(4).to_string(index=False))
    json.dump(head, open(RES / "headline.json", "w"), indent=1)
    print(json.dumps(head, indent=1))


if __name__ == "__main__":
    main()
