import importlib.util
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.multitest import multipletests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, G, MS = ROOT / "results", ROOT / "annotation/gold", ROOT / "manuscript"
(MS / "tables").mkdir(parents=True, exist_ok=True)
LOC = "C(locus, Treatment('dev_time'))[T.{}]"
AUT = "C(auth, Treatment('none'))[T.{}]"
GRS = ("filter", "constrained", "local", "human", "report", "optout", "provider", "none")
ROUND1 = "20261130-v1"


def num(n):
    return f"{int(round(n)):,}".replace(",", "{,}")


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def sgn(x, d=2):
    s = f"{x:.{d}f}"
    return s.replace("-", "$-$") if x < 0 else s


def pv(p):
    if p is None or np.isnan(p):
        return "--"
    return "$p<.001$" if p < .001 else f"$p={p:.3f}$".replace("=0.", "=.")


def main():
    H = json.load(open(RES / "headline.json"))
    M = pd.read_csv(RES / "models.csv")
    R = pd.read_csv(RES / "robustness.csv")
    SH = pd.read_csv(RES / "short_text.csv")
    mac = {}

    def coef(model, term, prefix, scale=1.0, frame=M):
        r = frame[(frame.model == model) & (frame.term == term)]
        if r.empty:
            return
        r = r.iloc[0]
        mac[prefix + "B"] = sgn(r.b * scale)
        mac[prefix + "SE"] = f"{r.se * scale:.2f}"
        mac[prefix + "Lo"], mac[prefix + "Hi"] = sgn(r.lo * scale), sgn(r.hi * scale)
        mac[prefix + "P"] = pv(r.p)
        if "p_holm" in r and not pd.isna(r.p_holm):
            mac[prefix + "PHolm"] = pv(r.p_holm)
        mac[prefix + "N"] = num(r.n)

    cc_p, idx = RES / "census_counts.json", ROOT / "data/raw/search_index.jsonl"
    if idx.exists() and "n_unreadable_agecheck" in H:
        cc = {"census_listings": sum(1 for _ in open(idx, encoding="utf-8")),
              "n_unreadable_agecheck": H["n_unreadable_agecheck"],
              "n_kept_not_app_page": H["n_kept_not_app_page"], "n_dropped_app_page": H["n_dropped_app_page"]}
        json.dump(cc, open(cc_p, "w"), indent=1)
    else:
        cc = json.load(open(cc_p))
    mac["CensusListings"] = num(cc["census_listings"])
    cohort = (ROOT / "data/processed/cohort_appids.txt").read_text().split()
    mac["CohortN"] = num(len(cohort))
    mac["PagesN"] = num(H["n_pages_all"])
    mac["PageErrors"] = num(len(cohort) - H["n_pages_all"])
    mac["ExcludedDates"] = num(H["n_excluded_dates"])
    mac["AnalysisN"] = num(H["n_cohort_with_page"])
    mac["NUnreadable"] = num(H["n_unreadable"])
    mac["NUnreadableAge"] = num(cc["n_unreadable_agecheck"])
    mac["ListedWindowN"] = num(H["n_listed_window"])
    assert cc["n_kept_not_app_page"] == cc["n_dropped_app_page"] == 0, "supplement says the two page tests agree"
    cs = H["code_src_reception"]
    mac["NGoldInRS"] = num(cs.get("gold_test", 0) + cs.get("gold_dev", 0))
    st = R[(R.model == "crit_term") & (R.term.str.startswith("C(auth") | (R.term == "ai_on_human"))].copy()
    st["p_holm"] = multipletests(st.p, method="holm")[1]
    srev = st[st.term.str.contains("review")].iloc[0]
    mac["StrictReviewB"], mac["StrictReviewPHolm"] = sgn(100 * srev.b), pv(srev.p_holm)
    saoh = st[st.term == "ai_on_human"].iloc[0]
    mac["StrictAohB"], mac["StrictAohP"], mac["StrictAohPHolm"] = sgn(100 * saoh.b), pv(saoh.p), pv(saoh.p_holm)
    mac["NDisclosed"] = num(H["n_disclosed"])
    mac["ShareDisclosed"] = pct(H["share_disclosed"])
    for y, w in (("2024", "TwentyFour"), ("2025", "TwentyFive"), ("2026", "TwentySix")):
        mac["ShareDisclosed" + w] = pct(H[f"share_disclosed_{y}"])
    mac["ShareDisclosedGeTen"] = pct(H["share_disclosed_ge10"])
    mac["ShareDisclosedLtTen"] = pct(H["share_disclosed_lt10"])
    mac["NUniqueDisclosures"] = num(H["n_unique_disclosures"])
    mac["ShareSharedText"] = pct(H["share_disclosures_shared_text"])
    mac["MedianWords"] = num(H["median_words"])
    mac["ShareSharedSameDev"] = pct(H["share_shared_same_dev"], 0)
    mac["ReceptionN"] = num(H["n_reception"])
    mac["ReceptionDisclosed"] = num(H["n_reception_disclosed"])
    mac["ReceptionDevs"] = num(H["n_reception_devs"])
    mac["ReceptionCoded"] = num(H["n_reception_disclosed_coded"])

    D = pd.read_csv(RES / "code_distribution.csv").set_index("code")["share"]
    names = {"pre": "Pre", "marketing": "Marketing", "marketing_only": "MarketingOnly",
             "explicit_no_live": "ExplicitNoLive", "ai_on_human": "AiOnHuman", "mixed_oversight": "Mixed",
             "d_visual": "Visual", "d_music": "Music", "d_sfx": "Sfx", "d_voice": "Voice", "d_text": "Text",
             "d_localization": "Localization", "d_code": "Code", "d_other": "Other", "scope_minor": "ScopeMinor",
             "scope_extensive": "ScopeExtensive", "placeholder": "Placeholder", "tools_named": "Tools",
             "provenance_claim": "Provenance", "small_team": "SmallTeam",
             "authority=0": "AuthNone", "authority=1": "AuthReview", "authority=2": "AuthEdit",
             "authority=3": "AuthFinal"}
    for k, v in names.items():
        if k in D.index:
            mac["Share" + v] = pct(D[k])
    mac["ShareAnyOversight"] = pct(1 - D["authority=0"])
    mac["ShareVisualAuthNone"] = pct(H["share_visual_auth0"])
    mac["NVisualAuthNone"] = num(H["n_visual_auth0"])
    mac["NVisualAuthNoneMkt"], mac["NVisualAuthNoneAoh"] = num(H["n_visual_auth0_mkt"]), num(H["n_visual_auth0_aoh"])
    mac["ShareVisualAuthNoneNoMkt"] = pct(H["share_visual_auth0_no_mkt"])
    mac["ShareVisualAuthNoneNoAoh"] = pct(H["share_visual_auth0_no_aoh"])
    mac["NAiOnHuman"], mac["NAuthNoneAoh"] = num(H["n_ai_on_human"]), num(H["n_auth0_aoh"])
    mac["ShareRuntimeReception"] = pct(H["n_live_reception"] / H["n_reception_disclosed_coded"])
    mac["NLiveReception"] = num(H["n_live_reception"])
    lc = pd.read_csv(RES / "locus_counts.csv", index_col=0)["n"]
    for k, v in (("dev_time", "DevTime"), ("runtime", "Runtime"), ("marketing_only", "MarketingOnly")):
        mac["NLocus" + v] = num(lc.get(k, 0))
    la = pd.read_csv(RES / "locus_by_authority.csv", index_col=0)
    for k, v in (("dev_time", "DevTime"), ("runtime", "Runtime"), ("marketing_only", "MarketingOnly")):
        mac["AuthNone" + v] = pct(la.loc[k, "none"], 0)
    assert mac["AuthNoneRuntime"] == mac["AuthNoneMarketingOnly"], "results.tex says '% of each'"

    gr = pd.read_csv(RES / "guardrails_live.csv", index_col=0)
    mac["NLiveCensus"] = num(H["n_live_census"])
    mac["NLiveCensusUnique"] = num(H["n_live_census_unique"])
    mac["NLiveSystem"] = num(H["n_live_system"])
    mac["NLivePlayer"] = num(H["n_live_player"])
    mac["ShareLiveAnyGuardrail"] = pct(H["live_any_guardrail"])
    mac["ShareGrSafety"] = pct(H["live_any_safety"])
    mac["NLiveInRS"], mac["NLiveOutRS"] = num(H["n_live_in_rs"]), num(H["n_live_out_rs"])
    mac["NLiveOutCue"] = num(H["live_out_cue_n"])
    mac["NLiveOutNoCue"] = num(H["n_live_out_rs"] - H["live_out_cue_n"])
    for k, v in (("in", "In"), ("out", "Out")):
        mac["ShareGrNone" + v] = pct(H[f"live_{k}_gr_none"])
        mac["ShareGrSafety" + v] = pct(H[f"live_{k}_any_safety"])
    for k, v in (("pre26", "Pre"), ("post26", "Post")):
        mac["ShareLiveOut" + v] = pct(H[f"n_live_out_{k}"] / (H[f"n_live_in_{k}"] + H[f"n_live_out_{k}"]), 0)
    if "share_rs" in gr:
        for k in gr.index:
            mac["ShareGrRS" + k.replace("c_gr_", "").capitalize()] = pct(gr.loc[k, "share_rs"])
    for k, v in (("pre26", "Pre"), ("post26", "Post"), ("in_pre26", "InPre"), ("in_post26", "InPost"),
                 ("out_post26", "OutPost")):
        mac["NLive" + v] = num(H[f"n_live_{k}"])
        mac["ShareGrNone" + v] = pct(H[f"live_{k}_gr_none"])
        mac["ShareGrSafety" + v] = pct(H[f"live_{k}_any_safety"])
    gt = pd.read_csv(ROOT / "data/processed/games.csv", low_memory=False,
                     usecols=["disclosure", "disclosed", "code_src", "page_redirect", "c_live_system", "c_live_player",
                              "c_gr_local", "c_gr_optout"])
    gt = gt[(gt.page_redirect.fillna(0) != 1) & (gt.disclosed == 1) & gt.code_src.notna() &
            ((gt.c_live_system == 1) | (gt.c_live_player == 1))]
    assert len(gt) == H["n_live_census"]
    on_device = (r"\blocal(?:ly)?\b|on[- ]device|offline|on (?:your|the player's) (?:own )?(?:machine|computer|pc|device|hardware)|"
                 r"your own (?:api|key|model)")
    mac["GrLocalNoDevice"] = num((~gt[gt.c_gr_local == 1].disclosure.str.contains(on_device, case=False)).sum())
    mac["GrOptoutRegen"] = num(gt[gt.c_gr_optout == 1].disclosure.str.contains(r"regenerat|re-?roll", case=False).sum())
    mac["CueRecallN"] = num(H["cue_recall_n"])
    mac["CueRecallNone"], mac["CueRecallGuard"] = pct(H["cue_recall_none"], 0), pct(H["cue_recall_guard"], 0)
    for k, v in (("filter", "Filter"), ("constrained", "Constrained"), ("local", "Local"), ("human", "Human"),
                 ("report", "Report"), ("optout", "Optout"), ("provider", "Provider"), ("none", "None")):
        mac["ShareGr" + v] = pct(gr.loc[f"c_gr_{k}", "share"])
        mac["NGr" + v] = num(gr.loc[f"c_gr_{k}", "n"])
        if not pd.isna(gr.loc[f"c_gr_{k}", "share_player"]):
            mac["ShareGrPlayer" + v] = pct(gr.loc[f"c_gr_{k}", "share_player"])

    coef("H1_pct_positive", "disclosed", "HOne")
    mac["RawPosDisc"] = f"{H['raw_pct_pos_disclosed']:.1f}"
    mac["RawPosNon"] = f"{H['raw_pct_pos_nondisclosed']:.1f}"
    r = M[(M.model == "H1_log_reviews_allwindow") & (M.term == "disclosed")].iloc[0]
    mac["HOneRevPct"] = sgn(100 * (np.exp(r.b) - 1), 1)
    mac["HOneRevLo"], mac["HOneRevHi"] = sgn(100 * (np.exp(r.lo) - 1), 1), sgn(100 * (np.exp(r.hi) - 1), 1)
    mac["HOneRevN"] = num(r.n)
    mac["HOneRevPctAbs"] = f"{abs(100 * (np.exp(r.b) - 1)):.1f}"
    mac["HOneAbs"] = f"{abs(M[(M.model == 'H1_pct_positive') & (M.term == 'disclosed')].b.iloc[0]):.1f}"
    r = M[(M.model == "within_crit_share") & (M.term == AUT.format("edit"))]
    if len(r):
        mac["WCEditLoAbs"] = f"{abs(100 * r.lo.iloc[0]):.2f}"
    mac["FENondiscN"], mac["FENondiscBefore"] = num(H["fe_nondisc_n"]), pct(H["fe_nondisc_before"], 0)
    for k, v in (("fe_devs", "FEDevs"), ("fe_switch_devs", "FESwitchDevs"), ("fe_switch_games", "FESwitchGames"),
                 ("fe_crit_switch_devs", "FECritSwitchDevs")):
        if k in H:
            mac[v] = num(H[k])
    if "n_text_games" in H:
        mac["TextGames"] = num(H["n_text_games"])
        mac["TextGamesDisc"] = num(H["n_text_games_disclosed"])
        mac["TextReviews"] = num(H["n_text_reviews"])
        for k, v in (("crit_share", "Crit"), ("mention_share", "Mention")):
            mac[v + "Disc"] = pct(H[f"{k}_disclosed"])
            mac[v + "Non"] = pct(H[f"{k}_nondisclosed"])
        mac["AnyCritDisc"] = pct(H["share_disclosed_any_crit"], 0)
        mac["AnyCritNon"] = pct(H["share_nondisclosed_any_crit"], 0)
        coef("H1_crit_share", "disclosed", "HOneCrit", 100)
        coef("H1_mention_share", "disclosed", "HOneMention", 100)
    if "n_text_games_disc_mention" in H:
        mac["TextGamesDiscMention"] = num(H["n_text_games_disc_mention"])
        for k, v in (("mention_share", "Mention"), ("crit_given_mention", "CritGiven")):
            mac[v + "DevTime"], mac[v + "Runtime"] = pct(H[f"{k}_dev_time"]), pct(H[f"{k}_runtime"])
        coef("crit_given_mention_weighted", LOC.format("runtime"), "WGWRuntime", 100, frame=R)
    coef("visual_no_marketing", "d_visual", "WVisualNoMkt", frame=R)
    coef("crit_visual_no_marketing", "d_visual", "WCVisualNoMkt", 100, frame=R)

    coef("multi_game_devs_pooled", "disclosed", "MGPooled", frame=R)
    coef("developer_fe", "disclosed", "DevFE", frame=R)
    coef("crit_share_multi_game_devs_pooled", "disclosed", "MGPooledCrit", 100, frame=R)
    coef("crit_share_developer_fe", "disclosed", "DevFECrit", 100, frame=R)
    coef("mention_share_developer_fe", "disclosed", "DevFEMention", 100, frame=R)
    coef("crit_term", "disclosed", "HOneCritTerm", 100, frame=R)
    coef("crit_share_term_developer_fe", "disclosed", "DevFECritTerm", 100, frame=R)
    for t, v in ((LOC.format("runtime"), "Runtime"), (LOC.format("marketing_only"), "MktOnly"), ("d_visual", "Visual"),
                 (AUT.format("review"), "Review"), ("ai_on_human", "AoH")):
        coef("crit_term", t, "WCT" + v, 100, frame=R)
    for k, v in (("crit_share_short", "Crit"), ("mention_share_short", "Mention"), ("crit_share_term_short", "CritTerm")):
        coef(f"{k}_developer_fe", "disclosed", "ShortFE" + v, 100, frame=SH)
    for t, v in ((LOC.format("runtime"), "Runtime"), (LOC.format("marketing_only"), "MktOnly"), ("d_visual", "Visual")):
        coef("crit_short", t, "ShortC" + v, 100, frame=SH)
    coef("crit_term_short", AUT.format("review"), "ShortStrictReview", 100, frame=SH)
    mac["ShortLong"], mac["ShortLongDisc"], mac["ShortLongNon"] = (pct(H[f"long_share_{x}"]) for x in ("all", "disc", "non"))
    mac["ShortLateCrit"], mac["ShortLateMention"] = pct(H["late_share_crit_all"]), pct(H["late_share_mention_all"])
    mac["NoTokenCrit"], mac["NoTokenCritShare"] = num(H["notoken_crit"]), pct(H["notoken_crit_share"])
    for full, short in (("within_crit_share", "crit_short"), ("within_mention_share", "mention_short"),
                        ("within_crit_given_mention", "crit_given_mention_short")):
        a = M[M.model == full].set_index("term")
        b = SH[SH.model == short].set_index("term").loc[a.index]
        tested = a.p_holm.notna()
        assert ((a.b > 0) == (b.b > 0)).all(), f"{full}: a sign changes at 1,500 characters; update Section S3"
        assert ((a.p_holm[tested] < .05) == (b.p_holm[tested] < .05)).all(), f"{full}: Holm status changes; update Section S3"
    mac["AoHN"], mac["AoHTranslCount"], mac["AoHRereleaseN"] = num(H["aoh_n"]), num(H["aoh_transl_n"]), num(H["aoh_rerelease_n"])
    coef("aoh_split", "aoh_transl", "AoHTransl", frame=R)
    coef("aoh_split", "aoh_other", "AoHOther", frame=R)
    coef("excl_rereleases", "ai_on_human", "AoHExclRerel", frame=R)
    mac["AoHPriorWorkN"], mac["DiscPriorWorkN"] = num(H["aoh_prior_work_n"]), num(H["disc_prior_work_n"])
    coef("excl_prior_work", "ai_on_human", "AoHExclPrior", frame=R)
    coef("prior_work_control", "ai_on_human", "AoHPriorCtrl", frame=R)
    mac["ShareDiscTrack"] = pct(H["share_disc_track"], 0)
    coef("track_record", "disclosed", "TRDisc", frame=R)
    for t, v in ((LOC.format("runtime"), "Runtime"), ("ai_on_human", "AoH"), ("d_visual", "Visual"),
                 (AUT.format("review"), "Review")):
        coef("track_record", t, "TR" + v, frame=R)
        coef("crit_track_record", t, "TRC" + v, 100, frame=R)
    mac["AccuseReviews"] = num(H["accuse_reviews"])
    mac["AccuseGamesDisc"], mac["AccuseGamesNon"] = num(H["accuse_games_disc"]), num(H["accuse_games_non"])
    mac["AccuseShareCrit"] = pct(H["accuse_share_of_crit_disc"])
    mac["FECritNoAccuseSwitchDevs"] = num(H["fe_crit_noaccuse_switch_devs"])
    coef("crit_noaccuse", "disclosed", "NoAcc", 100, frame=R)
    coef("crit_noaccuse_developer_fe", "disclosed", "NoAccFE", 100, frame=R)
    for y, w in (("2024", "TwentyFour"), ("2025", "TwentyFive"), ("2026", "TwentySix")):
        coef(f"crit_year_{y}", "disclosed", "CritYear" + w, 100, frame=R)
    if "wb_fe_n" in H:
        mac["WBFEEligN"], mac["WBFEExtraN"] = num(H["wb_fe_eligible"]), num(H["wb_fe_extra"])
        assert H["wb_fe_eligible_missing"] == 0, "supplement says every eligible game was targeted"
        for grp, v in (("fe", "FE"), ("random", "Rand")):
            mac[f"WB{v}N"] = num(H[f"wb_{grp}_n"])
            for stt in ("early", "late", "uncertain", "unknown"):
                mac[f"WB{v}{stt.capitalize()}"] = num(H[f"wb_{grp}_{stt}"])
        coef("crit_wb_nolate_developer_fe", "disclosed", "WBNoLateFE", 100, frame=R)
        coef("crit_wb_early_developer_fe", "disclosed", "WBEarlyFE", 100, frame=R)
        mac["WBEarlySwitchDevs"] = num(H["wb_early_switch_devs"])
    mac["ExpReviews"], mac["ExpCite"] = num(H["exp_reviews"]), pct(H["exp_cite"])
    mac["ExpCiteNone"], mac["ExpCiteOversight"] = pct(H["exp_cite_none"]), pct(H["exp_cite_oversight"])
    mac["ExpReviewsOversight"], mac["ExpOverRefOversight"] = num(H["exp_reviews_oversight"]), num(H["exp_overref_oversight"])
    FC = json.load(open(RES / "review_fetch_check.json"))
    assert FC["no_failed_fetch"] and FC["one_block_per_game"] and FC["ranks_consecutive"]
    mac["FetchGames"], mac["FetchRuns"] = num(FC["n_requested"]), "two three four five six seven eight nine".split()[FC["n_runs"] - 2]
    mac["FetchRepRows"], mac["FetchRepGames"] = num(FC["n_repeat_text_rows"]), num(FC["n_repeat_text_games"])
    LC = pd.read_csv(ROOT / "annotation/bulk/logic_corrections.csv")
    un = pd.read_csv(ROOT / "annotation/bulk/uid_map.csv").set_index("uid").norm
    mkt_norm = set(un[LC[(LC.code == "marketing_only") & (LC.corrected == 1)].uid])
    gl = pd.read_csv(ROOT / "data/processed/games.csv", low_memory=False,
                     usecols=["has_page", "page_redirect", "release_date", "review_count", "disclosed", "disclosure",
                              "code_src", "c_marketing_only"])
    gl = gl[(gl.has_page == 1) & (gl.page_redirect.fillna(0) != 1) & gl.release_date.between("2024-01-01", "2026-06-30")
            & (gl.review_count >= 10) & (gl.disclosed == 1) & (gl.code_src == "bulk")]
    moved = gl[gl.disclosure.str.strip().str.lower().str.replace(r"\s+", " ", regex=True).isin(mkt_norm)]
    assert (moved.c_marketing_only == 1).all()
    words = "zero one two three four five six seven eight nine".split()
    mac["LogicFixTexts"], mac["LogicFixMktRS"] = words[len(LC)], words[len(moved)]
    for k, v in (("ov_share", "OvShare"), ("ov_dom1", "OvDomOne"), ("ov_dom3", "OvDomThree"), ("ov_pre26", "OvPre"),
                 ("ov_post26", "OvPost")):
        mac[v] = pct(H[k], 0)
    mac["ShareSmallTeam"] = pct(H["share_small_team"])
    for t, v in (("n_domains", "Dom"), ("scope_minor", "ScopeMinor"), ("scope_extensive", "ScopeExt"), ("post26", "Post")):
        coef("oversight_extent", t, "OvExt" + v, 100)
    coef("log_reviews_multi_pooled", "disclosed", "RevMGPooled", frame=R)
    coef("log_reviews_developer_fe", "disclosed", "RevFE", frame=R)
    if "disp_pages" in H:
        mac["DispGames"], mac["DispPages"] = num(H["disp_games"]), num(H["disp_pages"])
        mac["DispPre"], mac["DispPost"] = num(H["disp_pre"]), num(H["disp_post"])
    sa = ROOT / "annotation/store_about"
    pg = pd.read_csv(sa / "pages.csv")
    cd = pd.read_csv(sa / "coded.csv", keep_default_na=False).set_index("appid")
    grc = ["filter", "constrained", "local", "human", "report", "optout", "provider"]
    gs = pd.read_csv(ROOT / "data/processed/games.csv", low_memory=False,
                     usecols=["appid", "c_gr_none"] + ["c_gr_" + c for c in grc]).set_index("appid").loc[cd.index]
    assert len(pg) == len(cd) == H["n_live_in_rs"] and set(pg.appid) == set(cd.index)
    page = cd.page_guardrails.str.split(";").map(lambda l: {x[3:] for x in l if x})
    disc = pd.Series([{c for c in grc if gs.loc[a, "c_gr_" + c] == 1} for a in cd.index], index=cd.index)
    none = gs.c_gr_none == 1
    has = page.map(len) > 0
    mac["AboutN"] = num(len(pg))
    mac["AboutDate"] = datetime.strptime(pg.fetched_at.max(), "%Y-%m-%d").strftime("%-d %B %Y")
    assert pg.fetched_at.min() == pg.fetched_at.max()
    mac["AboutOneAI"] = num(((pg.ai_headings == 1) & (pg.ai_headings_other == 0)).sum())
    mac["AboutDiscChanged"] = num(pg.disclosure_changed.sum())
    mac["AboutGrGames"] = num(has.sum())
    cand = pd.read_csv(sa / "candidates.csv")
    mac["AboutCandN"], mac["AboutCandGames"] = num(len(cand)), num(cand.appid.nunique())
    mac["AboutNoneN"] = num(none.sum())
    mac["AboutNoneFound"] = num((none & has).sum())
    mac["AboutNoneFoundLocal"] = num((none & page.map(lambda s: "local" in s)).sum())
    mac["AboutNoneFoundCue"] = num((none & has & (cd.screen.astype(str) == "1")).sum())
    mac["AboutCandCueN"] = num(cand.selected_by.str.contains("cue").sum())
    mac["AboutEarlyN"] = num(pg.has_early_access.sum())
    mac["AboutSubheadN"] = num((pg.ai_subheadings_about > 0).sum())
    rv = pd.read_csv(ROOT / "data/processed/games.csv", low_memory=False,
                     usecols=["appid", "review_count", "pct_positive"]).set_index("appid").loc[pg.appid]
    pr = pg.set_index("appid")
    allrow = pr.page_all_reviews.notna()
    mac["RateAllRowN"] = num(allrow.sum())
    mac["RateAllRowMedDiff"] = num(int((pr.page_all_reviews - rv.review_count)[allrow].abs().median()))
    mac["RateAllRowMedPct"] = num(int((pr.page_all_pct - rv.pct_positive)[allrow].abs().median()))
    en = ~allrow & pr.page_en_reviews.notna()
    assert (allrow | en).all()
    ratio = (rv.review_count / pr.page_en_reviews)[en]
    mac["RateEnOnlyN"] = num(en.sum())
    mac["RateEnOnlyRatioLo"], mac["RateEnOnlyRatioHi"] = f"{ratio.min():.1f}", f"{ratio.max():.1f}"
    mac["AboutNewType"] = num((~none & pd.Series([bool(p - d) for p, d in zip(page, disc)], index=cd.index)).sum())
    mac["ShareGrNonePage"] = pct((none & ~has).mean())
    mac["ShareGrNonePageN"] = num((none & ~has).sum())
    changed = pg.set_index("appid").disclosure_changed.loc[cd.index] == 1
    mac["AboutChangedNone"], mac["AboutChangedNoneFound"] = num((changed & none).sum()), num((changed & none & has).sum())
    rsc = pd.read_csv(ROOT / "annotation/runtime_scope/coded.csv").set_index("appid").loc[cd.index]
    judge = rsc.judge_only == 1
    assert len(rsc) == len(cd) and abs(none.mean() - H["live_in_gr_none"]) < 1e-9
    mac["RtJudgeN"], mac["RtJudgeNoneN"] = num(judge.sum()), num((judge & none).sum())
    mac["ShareGrNoneNoJudge"] = pct(none[~judge].mean())
    for k, v, sc in (("h2a_ingame_crit_share", "HTwoInGameCrit", 100), ("h2a_ingame_pct_positive", "HTwoInGamePos", 1)):
        c = H[k]
        mac[v + "B"], mac[v + "Lo"], mac[v + "Hi"] = sgn(c["b"] * sc), sgn(c["lo"] * sc), sgn(c["hi"] * sc)
        mac[v + "P"] = pv(c["p"])
    mac["RefN"] = num(H["h2a_ingame_pct_positive"]["n_ref"])
    mac["RefNoLocusN"] = num(H["h2a_ingame_pct_positive"]["n_ref_no_locus"])
    mac["RefNoLocusTextN"] = num(H["h2a_ingame_crit_share"]["n_ref_no_locus"])
    mac["RefTextN"] = num(H["h2a_ingame_crit_share"]["n_ref"])
    cw = R[R.model.isin(["cluster_dev_pub", "crit_cluster_dev_pub"])].copy()
    cw["base"] = [R[(R.model == ("main" if m == "cluster_dev_pub" else "crit_main")) & (R.term == t)].se.iloc[0]
                  for m, t in zip(cw.model, cw.term)]
    mac["TwoWaySEMax"] = pct((cw.se / cw.base - 1).abs().max(), 0)

    for model, pre, sc in (("within_pct_positive", "W", 1), ("within_crit_share", "WC", 100),
                           ("within_log_reviews", "WR", 1), ("within_mention_share", "WM", 100),
                           ("within_crit_given_mention", "WG", 100)):
        coef(model, LOC.format("runtime"), pre + "Runtime", sc)
        coef(model, LOC.format("marketing_only"), pre + "MktOnly", sc)
        coef(model, AUT.format("review"), pre + "Review", sc)
        coef(model, AUT.format("edit"), pre + "Edit", sc)
        coef(model, AUT.format("human_final"), pre + "Final", sc)
        for t, v in (("ai_on_human", "AoH"), ("d_visual", "Visual"), ("d_voice", "Voice"), ("d_music", "Music"),
                     ("d_text", "Text"), ("d_localization", "Loc"), ("d_code", "Code"), ("d_sfx", "Sfx"),
                     ("scope_minor", "ScopeMinor"), ("scope_extensive", "ScopeExt"), ("placeholder", "Placeholder")):
            coef(model, t, pre + v, sc)
    coef("within_pct_positive_anyoversight", "any_oversight", "WAnyOversight")
    coef("within_pct_positive_anyoversight_notext", "any_oversight", "WAnyOvNoText")
    coef("within_pct_positive_anyoversight_raw", "any_oversight", "WAnyOvRaw")

    ag_dev, ag_test = pd.read_csv(G / "agreement_dev.csv"), pd.read_csv(G / "agreement_test.csv")
    for a, v in ((ag_dev, "Dev"), (ag_test, "Test")):
        mac[v + "KappaMedian"] = f"{a.loc[a.code != 'authority', 'kappa'].median():.2f}"
        mac[v + "KappaMin"] = f"{a.loc[a.code != 'authority', 'kappa'].min():.2f}"
        mac[v + "KappaAuthority"] = f"{a.loc[a.code == 'authority', 'kappa'].iloc[0]:.2f}"
    mac["TestN"] = num(sum(1 for _ in open(G / "disclosures_test.jsonl")))
    mac["DevN"] = num(sum(1 for _ in open(G / "disclosures_dev.jsonl")))
    spec = importlib.util.spec_from_file_location("vc", Path(__file__).parent / "07_validate_coding.py")
    vc = importlib.util.module_from_spec(spec); spec.loader.exec_module(vc)
    ca, cb = ([json.loads(l) for l in open(G / f"coder{c}_disclosures_test.jsonl", encoding="utf-8")] for c in "AB")
    ks = vc.BIN + ["authority"]
    assert set(ks) == set(ca[0]) - {"gid"} and [r["gid"] for r in ca] == [r["gid"] for r in cb]
    CHUNK = 50
    dis = [sum(int(a[k]) != int(b[k]) for k in ks) for a, b in zip(ca, cb)]
    mac["TestChunkN"], mac["TestRestN"] = num(CHUNK), num(len(ca) - CHUNK)
    mac["TestChunkDis"], mac["TestChunkCells"] = num(sum(dis[:CHUNK])), num(CHUNK * len(ks))
    mac["TestRestDis"], mac["TestRestCells"] = num(sum(dis[CHUNK:])), num((len(ca) - CHUNK) * len(ks))
    kr = {}
    for k in ks:
        a_ = np.array([int(r[k]) for r in ca[CHUNK:]]); b_ = np.array([int(r[k]) for r in cb[CHUNK:]])
        kr[k] = (cohen_kappa_score(a_, b_, weights="quadratic" if k == "authority" else None)
                 if len(set(a_) | set(b_)) > 1 else np.nan)
    kb = pd.Series({k: v for k, v in kr.items() if k != "authority"})
    mac["TestRestKappaMedian"], mac["TestRestKappaMin"] = f"{kb.median():.2f}", f"{kb.min():.2f}"
    mac["TestRestKappaAuthority"] = f"{kr['authority']:.2f}"
    for sp, v in (("dev", "Dev"), ("test", "Test")):
        adj = [json.loads(l) for l in open(G / f"adjudicated_{sp}.jsonl", encoding="utf-8") if l.strip()]
        mac[v + "AdjItems"] = num(len(adj))
        mac[v + "AdjCells"] = num(sum(len([k for k in r if k not in ("gid", "note")]) for r in adj))
        mac[v + "AdjAoH"] = num(sum("ai_on_human" in r for r in adj))
    mac["RevAdjN"] = num(sum(1 for l in open(G / "adjudicated_reviews.jsonl", encoding="utf-8") if l.strip()))
    um = pd.read_csv(ROOT / "annotation/bulk/uid_map.csv")
    mac["BulkUnique"] = num(um.loc[um.uid != "gold", "uid"].nunique())
    mac["BulkGames"] = num(um.loc[um.uid != "gold", "appid"].nunique())
    mac["RuntimePassN"] = num(sum(1 for _ in open(ROOT / "annotation/bulk/runtime_pass.jsonl", encoding="utf-8")))
    final_p = G / "finalcoder_validation_test.csv"
    fv = pd.read_csv(final_p if final_p.exists() else G / "bulkcoder_validation_test.csv").set_index("code")
    rv = pd.read_csv(G / "rulecoder_validation_test.csv").set_index("code")
    mac["ModelAuthAcc"] = f"{fv.loc['authority (exact)', 'acc']:.2f}"
    mac["ModelAuthKappa"] = f"{fv.loc['authority (qw-kappa vs gold)', 'F1']:.2f}"
    mac["RuleAuthAcc"] = f"{rv.loc['authority (exact)', 'acc']:.2f}"
    mac["RuleAuthKappa"] = f"{rv.loc['authority (qw-kappa vs gold)', 'F1']:.2f}"
    rw = json.load(open(G / "review_validation.json"))
    mac["RevGoldN"] = num(rw["n"])
    mac["RevKappaMention"] = f"{rw['kappa_genai_mention']:.2f}"
    mac["RevKappaCrit"] = f"{rw['kappa_genai_critical']:.2f}"
    if "tone_n_runtime" in rw:
        mac["ToneRuntimeCrit"], mac["ToneRuntimeN"] = num(rw["tone_crit_runtime"]), num(rw["tone_n_runtime"])
        mac["ToneOtherPct"] = pct(rw["tone_crit_other"] / rw["tone_n_other"], 0)
        mac["ToneOtherCrit"], mac["ToneOtherN"] = num(rw["tone_crit_other"]), num(rw["tone_n_other"])
    for k, v in (("genai_mention", "Mention"), ("genai_critical", "Crit")):
        for s in ("P", "R", "F1"):
            mac[f"Lex{v}{s.replace('1', 'One')}"] = f"{rw[f'{k}_{s}_w']:.2f}"
        for grp, gv in (("disc", "Disc"), ("nondisc", "Non")):
            for s in ("P", "R"):
                mac[f"Lex{v}{s}{gv}"] = f"{rw[f'{k}_{s}_{grp}']:.2f}"
    for s in ("P", "R", "F1"):
        mac[f"LexCritTerm{s.replace('1', 'One')}"] = f"{rw[f'genai_critical_term_{s}_w']:.2f}"
    for k, v in (("genai_critical", "LexCrit"), ("genai_critical_term", "LexCritTerm")):
        mac[v + "PRuntime"], mac[v + "RRuntime"] = f"{rw[f'{k}_P_uruntime']:.2f}", f"{rw[f'{k}_R_uruntime']:.2f}"
    mac["LexRuntimeN"], mac["LexRuntimeNPos"] = num(rw["crit_n_runtime"]), num(rw["crit_npos_runtime"])
    for grp, gv in (("visual", "Visual"), ("notvisual", "NotVisual")):
        mac[f"LexCritP{gv}"], mac[f"LexCritR{gv}"] = (f"{rw[f'genai_critical_P_u{grp}']:.2f}",
                                                    f"{rw[f'genai_critical_R_u{grp}']:.2f}")
        mac[f"Lex{gv}N"] = num(rw[f"crit_n_{grp}"])
    mac["LexMktN"], mac["LexMktNPos"] = num(rw["crit_n_mktonly"]), num(rw["crit_npos_mktonly"])
    mac["LexRedirN"], mac["LexRedirGames"] = num(rw["redirect_n"]), num(rw["redirect_games"])
    mac["LexRedirCrit"] = num(rw["redirect_crit"])
    mac["LexNoTokenN"] = num(rw["no_token_n"])
    mac["LexNoTokenUpper"] = pct(rw["no_token_upper95"])
    mac["LexPoolN"] = num(rw["pop_token"] + rw["pop_no_token"])
    mac["NGuardTest"] = num(fv.loc["gr_none", "n"])
    mac["AoHGoldP"], mac["AoHGoldR"] = f"{fv.loc['ai_on_human', 'P']:.2f}", f"{fv.loc['ai_on_human', 'R']:.2f}"
    r = M[(M.model == "H1_log_reviews_allwindow") & (M.term == "disclosed")].iloc[0]
    mac["HOneLogRevB"] = sgn(r.b)

    hv = pd.read_csv(RES / "human_validation.csv")
    hj = json.load(open(RES / "human_validation.json"))
    hkey = pd.read_csv(ROOT / "annotation/human/items_key.csv")
    c1, c2 = hj["coders"][:2]

    def hval(t, comp, code, col="kappa"):
        r = hv[(hv.task == t) & (hv.comparison == comp) & (hv.code == code)]
        return r[col].iloc[0] if len(r) else np.nan

    def hh(t, code, col="kappa"):
        return hval(t, f"{c2} vs {c1}", code, col)

    def vs_h(t, pred, code, col="kappa"):
        return np.nanmean([hval(t, f"{pred} vs {c}", code, col) for c in (c1, c2)])

    mac["HumPN"] = num((hkey.task == "P").sum())
    for t, v in (("D", "D"), ("R", "R")):
        mac[f"Hum{v}N"] = num((hkey.task == t).sum())
        for s, n in hkey[hkey.task == t].stratum.value_counts().items():
            mac[f"Hum{v}" + {"live_cue": "Live", "rest": "Rest", "token": "Token", "random_no_token": "NoToken"}[s]] = num(n)
    core = [k for k in hv[hv.task == "D"].code.unique()
            if not k.startswith("gr_") and k not in ("live_any", "auth_none", "authority") and not np.isnan(hh("D", k))]
    mac["HumNCodes"] = num(len(core))
    mac["HumKappaHH"] = f"{np.nanmedian([hh('D', k) for k in core]):.2f}"
    mac["HumKappaMH"] = f"{np.nanmedian([vs_h('D', 'model', k) for k in core]):.2f}"
    mac["HumKappaGH"] = f"{np.nanmedian([vs_h('D', 'llm_gold', k) for k in core]):.2f}"
    mac["HumAuthHH"] = f"{hh('D', 'authority'):.2f}"
    mac["HumAuthMH"] = f"{vs_h('D', 'model', 'authority'):.2f}"
    mac["HumAoHHH"] = f"{hh('D', 'ai_on_human'):.2f}"
    mac["HumAoHMHMax"] = f"{max(hval('D', f'model vs {c}', 'ai_on_human') for c in (c1, c2)):.2f}"
    mac["HumAoHDis"] = num(hj["D_aoh_disagreements"])
    mac["HumAoHDisTransl"] = num(hj["D_aoh_disagreements_translation"])
    mac["HumAoHDisOther"] = num(hj["D_aoh_disagreements"] - hj["D_aoh_disagreements_translation"])
    assert hj["D_aoh_model_in_gold"] == hj["D_aoh_model_pos"]
    mac["AoHModelPos"], mac["AoHGoldPos"] = num(hj["D_aoh_model_pos"]), num(hj["D_aoh_gold_pos"])
    mac["AoHGoldExtra"], mac["AoHGoldExtraRej"] = num(hj["D_aoh_gold_extra"]), num(hj["D_aoh_gold_extra_rejected"])
    an = [hval("D", f"model vs {c}", "auth_none", "prev_ref_w") for c in (c1, c2)]
    mac["HumAuthNoneLo"], mac["HumAuthNoneHi"] = pct(min(an), 0), pct(max(an), 0)
    mac["HumAuthNoneModel"] = pct(hval("D", f"model vs {c1}", "auth_none", "prev_pred_w"), 0)
    for k, v in (("genai_mention", "Mention"), ("genai_critical", "Crit")):
        mac[f"HumRev{v}HH"] = f"{hh('R', k):.2f}"
        mac[f"HumLex{v}MH"] = f"{vs_h('R', 'model', k):.2f}"
        mac[f"HumLex{v}FOne"] = f"{vs_h('R', 'model', k, 'F1_w'):.2f}"
        mac[f"HumGold{v}MH"] = f"{vs_h('R', 'llm_gold', k):.2f}"

    hs = pd.read_csv(RES / "human_guardrail_shares.csv").set_index("code")
    g = hkey[hkey.task == "G"]
    mac["HumGN"] = num(len(g))
    mac["HumGRuntime"] = num((g.stratum == "runtime").sum())
    mac["HumGDecoy"] = num((g.stratum == "decoy").sum())
    mac["HumGGames"] = num(g[g.stratum == "runtime"].n_games.sum())
    mac["HumGDecoyCoded"] = num(sum(hj[f"G_decoys_coded_runtime_{c}"] for c in (c1, c2)))
    mac["HumGDis"] = num(hj["G_disagreement_cells"])
    gcodes = ["live_system", "live_player"] + [f"gr_{k}" for k in GRS]
    mac["HumGKappaHH"] = sgn(np.nanmedian([hh('G', k) for k in gcodes]))
    mac["HumGKappaMH"] = sgn(np.nanmedian([vs_h('G', 'model', k) for k in gcodes]))
    lg = [round(hs.loc["live_any", f"share_{c}"] * hs.loc["live_any", f"games_{c}"]) for c in (c1, c2)]
    mac["HumGLiveGamesLo"], mac["HumGLiveGamesHi"] = num(min(lg)), num(max(lg))
    for k, v in (("live_any", "Live"), ("live_system", "Sys"), ("live_player", "Player")):
        mac[f"HumG{v}HH"] = sgn(hh('G', k))
        mac[f"HumG{v}MH"] = sgn(vs_h('G', 'model', k))
    kh = [hh("G", f"gr_{k}") for k in GRS]
    mac["HumGrKappaHH"] = sgn(np.nanmedian(kh))
    mac["HumGrKappaHHLo"], mac["HumGrKappaHHHi"] = sgn(np.nanmin(kh)), sgn(np.nanmax(kh))
    mac["HumGrKappaMH"] = sgn(np.nanmedian([vs_h('G', 'model', f'gr_{k}') for k in GRS]))
    for k in GRS:
        v = k.capitalize()
        mac[f"HumGr{v}HH"] = sgn(hh('G', f'gr_{k}'))
        mac[f"HumGr{v}MH"] = sgn(vs_h('G', 'model', f'gr_{k}'))
        mac[f"ShareGrHum{v}"] = pct(hs.loc[f"gr_{k}", [f"share_{c}" for c in (c1, c2)]].mean())
    gdiff = {k: abs(100 * (hs.loc[f"gr_{k}", "share_model"] - hs.loc[f"gr_{k}", [f"share_{c}" for c in (c1, c2)]].mean()))
             for k in GRS}
    mac["HumGrShareMaxDiff"] = f"{max(gdiff.values()):.1f}"
    mac["HumGrShareMaxDiffOther"] = f"{max(v for k, v in gdiff.items() if k != 'constrained'):.1f}"
    hrs = pd.read_csv(RES / "human_guardrail_shares_rs.csv").set_index("source")
    assert hrs.games.eq(H["n_live_in_rs"]).all()
    for col, v in (("share_none", "None"), ("share_output_safeguard", "Safety")):
        x = hrs.loc[[c1, c2], col]
        mac[f"HumGr{v}InLo"], mac[f"HumGr{v}InHi"] = pct(x.min()), pct(x.max())
    r1 = RES / "human_round1"
    if (r1 / "human_validation.csv").exists():
        hv1 = pd.read_csv(r1 / "human_validation.csv")
        k1 = [hv1[(hv1.task == "G") & (hv1.comparison == f"{c2} vs {c1}") & (hv1.code == k)].kappa for k in gcodes]
        mac["HumGRoundOneKappaHH"] = sgn(np.nanmedian([x.iloc[0] for x in k1 if len(x)]))
        mac["HumGRoundOneDis"] = num(json.load(open(r1 / "human_validation.json"))["G_disagreement_cells"])
    hum_g = all(v.get("G") != ROUND1 for v in hj["versions"].values()) or bool(os.environ.get("HUMAN_G_PREVIEW"))

    S = pd.read_csv(RES / "supplement.csv")
    coef("mundlak_pct_positive", "mean_disclosed", "MundlakDiff", frame=S)
    coef("mundlak_crit_share", "mean_disclosed", "MundlakCritDiff", 100, frame=S)
    coef("crit_sub_pct_positive_pooled", "disclosed", "CritSubPooled", frame=S)
    coef("crit_sub_pct_positive_developer_fe", "disclosed", "CritSubFE", frame=S)
    coef("notags_H1_pct_positive", "disclosed", "NoTagsHOne", frame=S)
    coef("notags_H1_pct_positive_developer_fe", "disclosed", "NoTagsFE", frame=S)
    coef("notags_H1_crit_share", "disclosed", "NoTagsCrit", 100, frame=S)
    coef("notags_H1_crit_share_developer_fe", "disclosed", "NoTagsCritFE", 100, frame=S)
    for t, v in ((LOC.format("runtime"), "Runtime"), (LOC.format("marketing_only"), "MktOnly"), ("ai_on_human", "AoH"),
                 ("d_visual", "Visual"), ("d_localization", "Loc"), (AUT.format("edit"), "Edit")):
        coef("notags_within_pct_positive", t, "NoTagsW" + v, frame=S)
        coef("notags_within_crit_share", t, "NoTagsWC" + v, 100, frame=S)
    for t, v in (("d_localization", "Loc"), ("ai_on_human", "AoH"), ("d_music", "Music"), ("d_visual", "Visual"),
                 (LOC.format("runtime"), "Runtime")):
        coef("within_eng_pos", t, "Eng" + v, frame=S)
    coef("bulkonly_within_pct_positive", "ai_on_human", "BulkOnlyAoH", frame=S)
    if "bulkonly_n" in H:
        mac["BulkOnlyN"] = num(H["bulkonly_n"])
    assert H["label_test_bulk_outside_ref"] == 0
    assert H["label_test_ref_equals_both_annotators"] == H["label_test_games"]
    mac["AoHTestGames"], mac["AoHTestRef"], mac["AoHTestBulk"] = num(H["label_test_games"]), num(H["label_test_ref_pos"]), num(H["label_test_bulk_pos"])
    mac["AoHTestPosRefOnly"], mac["AoHTestPosNeither"], mac["AoHTestPosBoth"] = (f'{H["label_test_pos_" + k]:.1f}' for k in ("ref_only", "neither", "both"))
    LAB = {"Keep": "keepdev", "Zero": "devzero", "NoDev": "nodev", "AllBulk": "allbulk"}
    for v, k in LAB.items():
        coef(f"label_{k}", "ai_on_human", "Label" + v, frame=S)
        mac[f"Label{v}Holm"], mac[f"Label{v}MusicHolm"] = pv(H[f"label_{k}_holm"]), pv(H[f"label_{k}_music_holm"])
        mac[f"Label{v}Share"] = pct(H[f"label_{k}_share"])
        assert H[f"label_{k}_holm"] < .05
        mac[f"Label{v}VisualHolm"] = pv(H[f"label_{k}_crit_visual_holm"])
        kept = k in ("keepdev", "devzero")
        assert H[f"label_{k}_holm_changes"] == ([] if kept else ["d_music"]), H[f"label_{k}_holm_changes"]
        assert set(H[f"label_{k}_crit_holm_changes"]) <= (set() if kept else {"d_visual"}), H[f"label_{k}_crit_holm_changes"]
        assert H[f"label_{k}_crit_holm"] >= .05
    mac["NGoldDevInRS"] = num(H["code_src_reception"]["gold_dev"])
    lab_b = S[S.model.isin([f"label_{k}" for k in LAB.values()]) & (S.term == "ai_on_human")].b
    mac["LabelAoHMin"], mac["LabelAoHMax"] = sgn(lab_b.min()), sgn(lab_b.max())
    lab_sh = [H[f"label_{k}_share"] for k in LAB.values()]
    mac["LabelShareMin"], mac["LabelShareMax"] = pct(min(lab_sh)), pct(max(lab_sh))
    assert H["aoh_split_equal_p"] > .05 and H["label_nodev_split_equal_p"] > .05
    mac["AoHOtherCount"], mac["AoHSplitEqP"] = num(H["aoh_split_other_n"]), pv(H["aoh_split_equal_p"])
    coef("label_nodev_split", "aoh_transl", "LabelSplitTransl", frame=S)
    coef("label_nodev_split", "aoh_other", "LabelSplitOther", frame=S)
    mac["LabelSplitTranslCount"], mac["LabelSplitOtherCount"] = num(H["label_nodev_transl_n"]), num(H["label_nodev_other_n"])
    mac["LabelSplitEqP"] = pv(H["label_nodev_split_equal_p"])
    mac["DevMultiShareRS"], mac["DevMultiShareFE"] = pct(H["dev_multi_share_reception"]), pct(H["dev_multi_share_fe"])
    for u, ku in {"Single": "single", "Last": "last", "Linked": "linked"}.items():
        for o, ko, sc in (("Rate", "pct_positive", 1), ("Crit", "crit_share", 100), ("Strict", "crit_share_term", 100)):
            coef(f"devunit_{ku}_{ko}", "disclosed", f"FEUnit{u}{o}", sc, frame=S)
            r = S[(S.model == f"devunit_{ku}_{ko}") & (S.term == "disclosed")].iloc[0]
            assert (r.p < .05) == (o != "Rate")
    mac["LocN"], mac["LocAoHTranslN"] = num(H["loc_n"]), num(H["loc_aoh_transl_n"])
    mac["LowRevDiscN"], mac["LowRevDiscCoded"] = num(H["lowrev_disc_n"]), num(H["lowrev_disc_coded"])
    for k, v in (("reception", "Reception"), ("window", "Window"), ("within", "Within"), ("multi_game", "Multi"),
                 ("text", "Text"), ("within_text", "WithinText")):
        mac["Clusters" + v] = num(H["clusters"][k])
    mac["TopTags"] = ", ".join(t.replace("_", " ") for t in H["top_tags"])
    r2 = H["ov_r2_within"]
    lam, ppv, npv, a, lam_c = [], [], [], [], []
    for c in (c1, c2):
        r = hv[(hv.task == "D") & (hv.comparison == f"model vs {c}") & (hv.code == "auth_none")].iloc[0]
        tp = r.P_w * r.prev_pred_w; fp = r.prev_pred_w - tp; fn = r.prev_ref_w - tp; tn = 1 - tp - fp - fn
        ppv.append(tn / (1 - r.prev_pred_w)); npv.append(tp / (tp + fp)); lam.append(ppv[-1] + npv[-1] - 1)
        a.append(tn / (tn + fp) + tp / (tp + fn) - 1)
        lam_c.append((lam[-1] - r2 / a[-1]) / (1 - r2))
    lam, a, lam_c = float(np.mean(lam)), float(np.mean(a)), float(np.mean(lam_c))
    mac["OvPPV"], mac["OvNPV"] = f"{np.mean(ppv):.2f}", f"{np.mean(npv):.2f}"
    mac["OvLambda"], mac["OvLambdaC"], mac["OvRTwo"] = f"{lam:.2f}", f"{lam_c:.2f}", f"{r2:.2f}"
    mac["OvSensSpec"] = f"{a:.2f}"
    r = M[(M.model == "within_pct_positive_anyoversight") & (M.term == "any_oversight")].iloc[0]
    mac["WAnyOversightHiAdj"], mac["WAnyOversightLoAdj"] = sgn(r.hi / lam_c), sgn(r.lo / lam_c)
    cd, cn = H["crit_share_disclosed"], H["crit_share_nondisclosed"]
    cd_c = cd * rw["genai_critical_P_disc"] / rw["genai_critical_R_disc"]
    cn_c = cn * rw["genai_critical_P_nondisc"] / rw["genai_critical_R_nondisc"]
    mac["CritDiscCorr"], mac["CritNonCorr"] = pct(cd_c), pct(cn_c)
    mac["RawCritGap"], mac["RawCritGapCorr"] = pct(cd - cn), pct(cd_c - cn_c)
    for grp, gv in (("disc", "Disc"), ("nondisc", "Non")):
        mac["LexCritN" + gv] = num(rw[f"genai_critical_n_{grp}"])
    mac["LexCritNPosNon"] = num(rw["crit_npos_nondisc"])
    mac["LexCritNPosDisc"] = num(rw["crit_npos_runtime"] + rw["crit_npos_otherdisc"])
    rm = pd.read_csv(G / "review_meta.csv")
    mac["LexTokenN"] = num((rm.stratum == "token").sum())
    mac["LexPopToken"], mac["LexPopNoToken"] = num(rw["pop_token"]), num(rw["pop_no_token"])
    ra_ = pd.read_csv(RES / "rerun_agreement.csv")
    s_ = ra_[~ra_.convention_changed & (ra_.code != "authority") & (ra_[["n_pos_v1", "n_pos_v2"]].min(axis=1) >= 5)]
    mac["RerunN"] = num(ra_.n.iloc[0])
    mac["RerunKappaMedian"], mac["RerunKappaMin"] = f"{s_.kappa.median():.2f}", f"{s_.kappa.min():.2f}"
    mac["RerunAuthKappa"] = f"{ra_.set_index('code').loc['authority', 'kappa']:.2f}"
    mac["RerunMaxDis"] = pct(1 - ra_[~ra_.convention_changed].agree.min())
    def vis_mkt(p):
        rs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        return np.mean([r["d_visual"] for r in rs if r.get("marketing_only") == 1])
    vm = [vis_mkt(p) for p in sorted((ROOT / "annotation/bulk").glob("coded_batch_[0-9][0-9].jsonl"))]
    mac["MktVisualMin"], mac["MktVisualMax"] = pct(min(vm), 0), pct(max(vm), 0)
    mac["MktVisualFirst"] = pct(vis_mkt(ROOT / "annotation/bulk/coded_batch_02_v1.jsonl"), 0)

    PJ = json.load(open(RES / "publisher.json"))
    P = pd.read_csv(RES / "publisher.csv")
    mac["PubN"], mac["PubShare"] = num(PJ["n_est"]), pct(PJ["share_est"])
    mac["PubSepN"], mac["PubSepShare"] = num(PJ["n_sep"]), pct(PJ["share_sep"])
    mac["PubPublishers"] = num(PJ["n_est_publishers_cohort"])
    mac["PubDiscEst"], mac["PubDiscSelf"] = pct(PJ["disc_est"]), pct(PJ["disc_self"])
    mac["PubRdN"] = num(PJ["n_rd_est"])
    mac["PubOvEst"], mac["PubOvSelf"] = pct(PJ["any_oversight_est"]), pct(PJ["any_oversight_self"])
    mac["PubFEDevs"], mac["PubFEGames"] = num(PJ["n_fe_devs_both"]), num(PJ["n_fe_games_both"])
    mac["PubDevsTwo"], mac["PubDevsReuse"] = num(PJ["n_devs_2disc"]), num(PJ["n_devs_2disc_reuse"])
    mac["PubDevsReuseShare"] = pct(PJ["n_devs_2disc_reuse"] / PJ["n_devs_2disc"], 0)
    mac["PubPubsTwo"], mac["PubPubsShared"] = num(PJ["n_pubs_2dev"]), num(PJ["n_pubs_2dev_shared"])
    mac["PubGamesShared"], mac["PubRuntimeN"] = num(PJ["n_games_pub_shared"]), num(PJ["n_runtime_est"])
    coef("disclosed", "pub_est", "PubDisc", 100, P)
    coef("devfe_disclosed", "pub_est", "PubFE", 100, P)
    coef("devfe_nocov_disclosed", "pub_est", "PubFENoCov", 100, P)
    coef("broad_disclosed", "pub_sep", "PubBroadDisc", 100, P)
    coef("est5_disclosed", "pub_est5", "PubFiveDisc", 100, P)
    for y, k in (("any_oversight", "Ov"), ("live_any", "Rt"), ("marketing_only", "Mkt")):
        coef(y, "pub_est", "Pub" + k, 100, P)
    for y, k, s in (("pct_positive", "Rate", 1), ("crit_share", "Crit", 100)):
        coef(y, "disclosed:pub_est", "PubInt" + k, s, P)
        coef(y, "disclosed", "PubGapSelf" + k, s, P)
        coef(y, "disclosed_if_publisher", "PubGapPub" + k, s, P)
        coef(y, "publisher_if_disclosed", "PubLvl" + k, s, P)
    mac["PubAnyN"], mac["PubCoN"], mac["PubCoDisc"] = num(PJ["n_est_any"]), num(PJ["n_est_co"]), pct(PJ["disc_est_co"])
    coef("any_disclosed", "pub_est_any", "PubAnyDisc", 100, P)
    coef("other_disclosed", "pub_est_other", "PubOtherDisc", 100, P)
    mac["PubNotOtherN"], mac["PubNotOtherShare"] = num(PJ["n_est_not_other"]), pct(PJ["n_est_not_other"] / PJ["n_est"])
    mac["PubNotOtherDisc"] = pct(PJ["disc_est_not_other"])
    CK = pd.read_csv(RES / "publisher_check_sample.csv", dtype=str)
    drawn_est = CK.drawn_from.str.startswith("established")
    est_now, sep_now = CK.group_now == "established", CK.group_now == "separate, not established"
    mac["PubCheckTotal"], mac["PubCheckDrawn"] = num(len(CK)), num(drawn_est.sum())
    mac["PubCheckSepDrawn"] = num((~drawn_est).sum())
    mac["PubCheckCo"] = num((drawn_est & CK.group_now.str.startswith("co-published")).sum())
    mac["PubCheckEstN"] = num(est_now.sum())
    mac["PubCheckEstDistinct"] = num((est_now & (CK.final == "distinct")).sum())
    mac["PubCheckEstOwn"] = num((est_now & (CK.final == "own")).sum())
    mac["PubCheckSepN"] = num(sep_now.sum())
    mac["PubCheckSepDistinct"] = num((sep_now & (CK.final == "distinct")).sum())
    mac["PubCheckAgree"] = num((CK.ai_judgment == CK.author_label).sum())
    mac["PubCheckRevised"] = num((CK.author_label_initial != CK.author_label).sum())
    mac["PubCheckAgreeInitial"] = num((CK.ai_judgment == CK.author_label_initial).sum())
    for k, model in (("PubDisc", "disclosed"), ("PubFE", "devfe_disclosed")):
        mac[k + "Abs"] = f"{abs(100 * P[(P.model == model) & (P.term == 'pub_est')].b.iloc[0]):.1f}"
    rq3 = P[(P.model.isin(["any_oversight", "live_any", "marketing_only", "pct_positive", "crit_share"]))
            & P.p_holm.notna()]
    mac["PubMinPHolm"] = pv(rq3.p_holm.min())
    mac["PubMinP"] = pv(rq3.p.min())
    alt = P[P.model.str.match(r"(broad|est5|devcl|any|other)_") & ~P.model.str.endswith("_disclosed") & P.p_holm.notna()]
    mac["PubAltMinPHolm"] = pv(alt.p_holm.min())
    mac["PubLvlMinP"] = pv(P[P.term == "publisher_if_disclosed"].p.min())

    with open(MS / "numbers.tex", "w", encoding="utf-8") as f:
        f.write("% generated by code/15_numbers.py -- do not edit\n")
        for k, v in mac.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

    rows = [("pre", "Development-time content"), ("live_any", "Runtime generation (any)$^c$"),
            ("live_system", "Runtime, system-initiated$^c$"),
            ("live_player", "Runtime, player-initiated$^c$"), ("marketing_only", "Marketing only"),
            ("ai_on_human", "AI applied to human work"),
            ("d_visual", "Visual"), ("d_music", "Music"), ("d_voice", "Voice"), ("d_text", "Text"),
            ("d_localization", "Localization"), ("d_code", "Code"), ("scope_minor", "Limited scope")]
    a = ag_test.set_index("code")
    ca = {r["gid"]: r for r in map(json.loads, open(G / "coderA_disclosures_test.jsonl"))}
    cb = {r["gid"]: r for r in map(json.loads, open(G / "coderB_disclosures_test.jsonl"))}
    la = [int(ca[g]["live_system"] or ca[g]["live_player"]) for g in ca]
    lb = [int(cb[g]["live_system"] or cb[g]["live_player"]) for g in ca]
    a.loc["live_any", "kappa"] = cohen_kappa_score(la, lb)
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{Coding validation on the test sample (" +
             mac["TestN"] + r" disclosures) and the review sample (" + mac["RevGoldN"] + r" reviews)}",
             r"\label{tab:validation}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{0.92}",
             r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"Code & Prev. & $\kappa_{H}$ & $\kappa_{MH}$ & $F_1^{m}$ & $F_1^{r}$ \\", r"\midrule",
             r"\multicolumn{6}{@{}l}{\emph{Disclosures}} \\"]
    for k, lab in rows:
        lines.append(f"{lab} & {pct(fv.loc[k, 'gold_prev'])} & {hh('D', k):.2f} & "
                     f"{vs_h('D', 'model', k):.2f} & {fv.loc[k, 'F1']:.2f} & {rv.loc[k, 'F1']:.2f} \\\\")
    lines.append(f"Authority$^b$ & -- & "
                 f"{hh('D', 'authority'):.2f}$^a$ & {vs_h('D', 'model', 'authority'):.2f}$^a$ & "
                 f"{fv.loc['authority (exact)', 'acc']:.2f} & {rv.loc['authority (exact)', 'acc']:.2f} \\\\")
    lines += [r"\multicolumn{6}{@{}l}{\emph{Reviews}} \\",
              f"Refers to generative AI & {pct(rw['genai_mention_gold_prev_w'])} & "
              f"{hh('R', 'genai_mention'):.2f} & {vs_h('R', 'model', 'genai_mention'):.2f} & "
              f"-- & {rw['genai_mention_F1_w']:.2f} \\\\",
              f"Criticizes generative AI & {pct(rw['genai_critical_gold_prev_w'])} & "
              f"{hh('R', 'genai_critical'):.2f} & {vs_h('R', 'model', 'genai_critical'):.2f} & "
              f"-- & {rw['genai_critical_F1_w']:.2f} \\\\",
              r"\bottomrule", r"\end{tabular}",
              r"\par\smallskip\parbox{\columnwidth}{\scriptsize Prev.: prevalence (\%) in the model-adjudicated "
              r"reference labels. Prevalence and $F_1$ values are weighted to each sample's target population. " +
              (r"Guardrail codes: Table~\ref{tab:guardrails}. " if hum_g else r"Guardrail codes (not shown) were "
               r"validated on the " + mac["NGuardTest"] + r" runtime texts of the test sample. ") +
              r"$\kappa$ columns use the " + mac["HumDN"] + r" disclosures and " + mac["HumRN"] + r" reviews coded by both "
              r"human coders, $F_1$ columns the full samples. "
              r"$\kappa_{H}$: Cohen's $\kappa$ ($^a$quadratic-weighted) between the two human coders; $\kappa_{MH}$: final codes (bulk codes after the runtime re-coding; reviews: "
              r"lexicon) against each coder, averaged. $F_1^{m}$: final codes against the reference labels. $F_1^{r}$: "
              r"rule-based coder, a reproducible baseline (reviews: lexicon). "
              r"$^b$Highest level claimed; $F_1$ columns give exact accuracy. $^c$After the runtime re-coding, which was "
              r"designed after inspecting this sample; Supplementary Table~S4 gives the independent human check. "
              r"Model-annotator agreement and all codes: Supplementary Table~S3.}",
              r"\end{table}"]
    (MS / "tables/tab_validation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    spec = [("main", "Main specification"), ("min_reviews_5", r"$\geq$5 reviews"), ("min_reviews_20", r"$\geq$20 reviews"), ("min_reviews_50", r"$\geq$50 reviews"),
            ("no_early_access", "Excluding early access"), ("excl_prolific_devs_gt5", r"Developers with $\leq$5 games"),
            ("year_2024", "Released 2024"), ("year_2025", "Released 2025"), ("year_2026h1", "Released Jan.--Jun. 2026"),
            ("track_record", "Developer track record$^f$"), ("excl_prior_work", "Excluding prior releases, series$^g$"),
            ("rule_codes", "Rule-based codes"), ("fractional_logit", "Fractional logit$^a$"),
            ("multi_game_devs_pooled", "Multi-game developers, pooled"), ("developer_fe", "Developer fixed effects$^d$")]
    fv = H["fe_disc_vary"]
    vlo, vhi = min(fv, key=fv.get), max(fv, key=fv.get)
    vlab = {"runtime": "runtime", "marketing_only": "marketing only", "review": "review", "edit": "edit",
            "human_final": "human final", "ai_on_human": "AI on human", "d_visual": "visual"}
    cols = [("disclosed", "Disclosed"), (LOC.format("runtime"), "Runtime"), (LOC.format("marketing_only"), "Mkt. only"),
            (AUT.format("review"), "Review"), (AUT.format("edit"), "Edit"), (AUT.format("human_final"), "Human final"), ("ai_on_human", "AI on human"),
            ("d_visual", "Visual")]

    Wm = M.set_index(["model", "term"])

    def cell(r, sc=1, main=None):
        ph = Wm.p_holm.get((main, r.name), np.nan) if main else np.nan
        star = r"^{\dagger}" if not pd.isna(ph) and ph < .05 else ("^{*}" if r.p < .05 else "")
        b = r.b * sc
        return f"${b:.2f}{star}$" if b >= 0 else f"$-{abs(b):.2f}{star}$"

    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Robustness of the main estimates}",
             r"\label{tab:robust}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{0.9}",
             r"\begin{tabular}{@{}l" + "r" * len(cols) + "rr@{}}", r"\toprule",
             " & ".join(["Specification"] + [c[1] for c in cols] + [r"$N$ (all)", r"$N$ (discl.)"]) + r" \\",
             r"\midrule", r"\multicolumn{" + str(len(cols) + 3) + r"}{@{}l}{\emph{A. Percentage of positive reviews}} \\"]
    crit_spec = [("crit_main", "Main specification"), ("crit_mention_share", r"\quad Refers to generative AI$^c$"),
                 ("crit_given_mention", r"\quad Critical, given a reference$^c$"), ("crit_min_reviews_20", r"$\geq$20 reviews"),
                 ("crit_term", "Critical term required$^e$"), ("crit_short", "First 1,500 characters$^h$"),
                 ("crit_track_record", "Developer track record$^f$"),
                 ("crit_rule_codes", "Rule-based codes"),
                 ("crit_share_developer_fe", "Developer fixed effects$^b$")]
    for key, lab in spec + [(None, None)] + crit_spec:
        if key is None:
            lines += [r"\midrule", r"\multicolumn{" + str(len(cols) + 3) + r"}{@{}l}{\emph{B. Share of reviews "
                      r"flagged as critical of generative AI (percentage points)}} \\"]
            continue
        sc = 100 if key.startswith("crit") else 1
        rr = pd.concat([R, SH])
        rr = rr[rr.model == key].set_index("term")
        if rr.empty:
            continue
        main = {"main": "within_pct_positive", "crit_main": "within_crit_share"}.get(key)
        cells = [cell(rr.loc[t], sc, main if t != "disclosed" else None) if t in rr.index else "--" for t, _ in cols]
        n_all = num(rr.loc["disclosed", "n"]) if "disclosed" in rr.index else "--"
        n_d = num(rr.loc[cols[1][0], "n"]) if cols[1][0] in rr.index else "--"
        lines.append(" & ".join([lab] + cells + [n_all, n_d]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\par\smallskip\parbox{\textwidth}{\scriptsize OLS with developer-clustered standard "
              r"errors and all covariates. Disclosed (H1): all games in the reception sample (row "
              r"$\geq$5 reviews: all window games with at least five reviews); other columns: "
              r"disclosing games (references: development-time locus, no authority stated); row $\geq$5 reviews adds "
              r"only the " + mac["LowRevDiscCoded"] + r" of " + mac["LowRevDiscN"] + r" disclosing games with 5--9 "
              r"reviews whose text was coded (runtime-cue and reference-sample texts), so it over-represents runtime "
              r"generation. $^{\dagger}$Holm-adjusted "
              r"$p<.05$ (main row, as in Table~\ref{tab:within}); $^{*}p<.05$, unadjusted. $^a$Log-odds. $^b$H1 only, multi-game-developer subsample. "
              r"$^c$Components of the main specification: share of reviews referring to generative AI, and share of "
              r"these references criticizing it (games with a reference). $^d$Other columns: disclosing games of "
              f"{num(H['fe_disc_devs'])} developers, of which only {fv[vlo]} ({vlab[vlo]}) to {fv[vhi]} ({vlab[vhi]}) "
              f"vary in a given code, so content estimates here are barely identified. $^e$Requires a critical term in a "
              f"referring sentence, ignoring the recommendation. $^f$Adds the developer's mean positive-review share on "
              f"other games (known for {mac['ShareDiscTrack']}\\% of disclosing games) and a missing indicator. "
              f"$^g$Excludes remasters, ports, re-releases, and series entries. $^h$Lexicon limited to the first 1,500 "
              f"characters of each review, the length of the validation sample.}}",
              r"\end{table*}"]
    (MS / "tables/tab_robust.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    gdef = [("constrained", "Constrained scope", "Prompt, preset, or topic limits"),
            ("local", "Local or private", "On-device, offline, or data privacy"),
            ("filter", "Filtering", "Input or output moderation"),
            ("optout", "Player opt-out", "Disable, opt in, or regenerate"),
            ("report", "Reporting", "Channel to report output"),
            ("human", "Human monitoring", "Developers monitor output"),
            ("provider", "Provider policies", "Relies on vendor safeguards"),
            ("none", "None stated", "No guardrail mentioned")]
    ex_p = RES / "guardrail_examples.csv"
    ex = pd.read_csv(ex_p).set_index("code")["example"].to_dict() if ex_p.exists() and not hum_g else {}
    hcols = [("ShareGrRS{}", r"\%$_{R}$"), ("ShareGrHum{}", r"\%$_{H}$")] if hum_g else []
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Guardrails stated in runtime disclosures (" + mac["NLiveCensus"] + r" games)}",
             r"\label{tab:guardrails}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}p{" + ("0.25" if hum_g else "0.21") + r"\columnwidth}" + ("r" * (1 + len(hcols)) if hum_g else "rr") +
             r">{\raggedright\arraybackslash}p{" + ("0.46" if hum_g else "0.58") + r"\columnwidth}@{}}", r"\toprule",
             r"Guardrail & " + ("" if hum_g else r"$n$ & ") + r"\% & " + "".join(h + " & " for _, h in hcols) +
             ("Definition" if hum_g else "Definition and example") + r" \\", r"\midrule"]
    for k, lab, dfn in gdef:
        e = ex.get(k)
        cell = dfn + (r". \emph{``" + e + r"''}" if isinstance(e, str) and e else "")
        hv_ = "".join(mac[m.format(k.capitalize())].replace("nan", "--") + " & " for m, _ in hcols)
        n_ = "" if hum_g else mac["NGr" + k.capitalize()] + " & "
        lines.append(f"{lab} & {n_}{mac['ShareGr' + k.capitalize()]} & {hv_}{cell} \\\\")
    hnote = (r" \%: final codes of all " + mac["NLiveCensus"] + r" games."
             r" \%$_{R}$: among the " + mac["NLiveInRS"] + r" runtime disclosures of the reception sample, all of which "
             r"were coded." + r" \%$_{H}$: among all " + mac["HumGGames"] + r" games, as coded by the two human coders (mean); agreement: Supplementary "
             r"Table~S4.") if hum_g else ""
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\par\smallskip\parbox{\columnwidth}{\scriptsize A disclosure can state several guardrails." +
              ("" if hum_g else " Examples are quoted verbatim from store pages.") + hnote + "}", r"\end{table}"]
    (MS / "tables/tab_guardrails.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows = [(None, "Locus (ref.: development time)"), (LOC.format("runtime"), "Runtime generation"),
            (LOC.format("marketing_only"), "Marketing only"), (None, "Stated authority (ref.: none)"),
            (AUT.format("review"), "Review or selection"), (AUT.format("edit"), "Editing"),
            (AUT.format("human_final"), "Human-made final content"), (None, "Provenance (ref.: not flagged)"),
            ("ai_on_human", "AI applied to human work"), (None, "Content domain"), ("d_visual", "Visual"), ("d_voice", "Voice"), ("d_music", "Music"),
            ("d_sfx", "Sound effects"), ("d_text", "Text"), ("d_localization", "Localization"), ("d_code", "Code"),
            (None, "Framing and other codes"), ("scope_minor", "Limited scope"), ("scope_extensive", "Pervasive use"),
            ("placeholder", "Placeholder"), ("tools_named", "Tools named"), ("small_team", "Small team"),
            ("explicit_no_live", "No runtime generation stated"), ("log_words", "Log disclosure length")]
    models = [("within_pct_positive", 1), ("within_crit_share", 100)]
    W = M.set_index(["model", "term"])

    def wcell(model, t, sc):
        if (model, t) not in W.index:
            return "--", ""
        r = W.loc[(model, t)]
        mark = r"$^{\dagger}$" if not pd.isna(r.get("p_holm", np.nan)) and r.p_holm < .05 else \
            (r"$^{*}$" if r.p < .05 else "")
        return sgn(r.b * sc) + mark, f"({r.se * sc:.2f})"

    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Disclosure content and reception among disclosing games (OLS)}", r"\label{tab:within}",
             r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{0.92}",
             r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{Positive reviews} & \multicolumn{2}{c}{AI-critical reviews} \\",
             r"\cmidrule(lr){2-3}\cmidrule(l){4-5}", r" & $b$ & (SE) & $b$ & (SE) \\", r"\midrule"]
    for t, lab in rows:
        if t is None:
            lines.append(r"\multicolumn{5}{@{}l}{\emph{" + lab + r"}} \\")
            continue
        c = [x for model, sc in models for x in wcell(model, t, sc)]
        lines.append(f"\\quad {lab} & " + " & ".join(c) + r" \\")
    nn = [num(M[M.model == m].n.iloc[0]) if (M.model == m).any() else "--" for m, _ in models]
    r2 = [f"{M[M.model == m].r2.iloc[0]:.2f}" if (M.model == m).any() else "--" for m, _ in models]
    lines += [r"\midrule", f"Games & \\multicolumn{{2}}{{r}}{{{nn[0]}}} & \\multicolumn{{2}}{{r}}{{{nn[1]}}} \\\\",
              f"$R^2$ & \\multicolumn{{2}}{{r}}{{{r2[0]}}} & \\multicolumn{{2}}{{r}}{{{r2[1]}}} \\\\",
              r"\bottomrule", r"\end{tabular}",
              r"\par\smallskip\parbox{\columnwidth}{\scriptsize Coefficients in percentage points with "
              r"developer-clustered standard errors. Both models include release-month fixed effects, the 30 most "
              r"frequent tags, and all covariates listed in Section~\ref{sec:methods}. AI-critical reviews: games with "
              r"at least five collected reviews. $^{\dagger}$Holm-adjusted $p<.05$ within the hypothesis family; "
              r"$^{*}p<.05$, unadjusted (terms outside the families, or not significant after adjustment).}", r"\end{table}"]
    (MS / "tables/tab_within.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    clab = {"pre": "Development-time content", "live_system": "Runtime, system-initiated",
            "live_player": "Runtime, player-initiated", "live_any": "Runtime (any)", "marketing": "Marketing material",
            "marketing_only": "Marketing only", "explicit_no_live": "No runtime generation stated",
            "non_genai": "No generative AI", "auth_none": "No human oversight stated", "ai_on_human": "AI applied to human work",
            "mixed_oversight": "Mixed oversight", "d_visual": "Visual", "d_music": "Music", "d_sfx": "Sound effects",
            "d_voice": "Voice", "d_text": "Text", "d_localization": "Localization", "d_code": "Code",
            "d_other": "Other uses", "scope_minor": "Limited scope", "scope_extensive": "Pervasive use",
            "placeholder": "Placeholder", "tools_named": "Tools named", "provenance_claim": "Provenance claim",
            "small_team": "Small team", "authority": "Authority$^a$"}
    modelled = {"live_system", "live_player", "marketing_only", "ai_on_human", "d_visual", "d_music", "d_sfx", "d_voice",
                "d_text", "d_localization", "d_code", "scope_minor", "scope_extensive", "placeholder", "tools_named",
                "small_team", "explicit_no_live", "authority"}
    ag = ag_test.set_index("code").kappa.to_dict()
    ag["live_any"] = a.loc["live_any", "kappa"]
    rr_ = ra_.set_index("code").kappa.to_dict()
    f2 = lambda x: "--" if x is None or pd.isna(x) else f"{x:.2f}"
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{Agreement for every disclosure code}", r"\label{tab:codes}",
             r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{0.92}",
             r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"Code & \% & $n_{+}$ & $\kappa_{AB}$ & $\kappa_{H}$ & $\kappa_{MH}$ & $\kappa_{RR}$ \\", r"\midrule"]
    for k in ["pre", "live_any", "live_system", "live_player", "marketing", "marketing_only", "explicit_no_live",
              "non_genai", "auth_none", "authority", "ai_on_human", "mixed_oversight", "d_visual", "d_music", "d_sfx",
              "d_voice", "d_text", "d_localization", "d_code", "d_other", "scope_minor", "scope_extensive", "placeholder",
              "tools_named", "provenance_claim", "small_team"]:
        share = {"auth_none": D.get("authority=0"), "authority": None, "live_any": H["n_live_reception"] /
                 H["n_reception_disclosed_coded"]}.get(k, D.get(k))
        npos = np.nanmean([hval("D", f"model vs {c}", k, "n_pos_ref") for c in (c1, c2)]) if k != "authority" else np.nan
        lab = clab[k] + (r"$^{m}$" if k in modelled else "")
        lines.append(f"{lab} & {'--' if share is None else pct(share)} & {'--' if pd.isna(npos) else num(npos)} & "
                     f"{f2(ag.get(k))} & {f2(hh('D', k))} & {f2(vs_h('D', 'model', k))} & {f2(rr_.get(k))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\par\smallskip\parbox{\columnwidth}{\scriptsize \%: share of coded disclosures in the reception sample. "
              r"$n_{+}$: positives in the human-coded subsample of " + mac["HumDN"] + r" test disclosures (mean of the "
              r"two coders); $\kappa$ is unstable below about five. $\kappa_{AB}$: two model annotators on the " +
              mac["TestN"] + r" test disclosures before adjudication; $\kappa_{H}$: the two human coders; "
              r"$\kappa_{MH}$: final codes against each coder, averaged; $\kappa_{RR}$: the bulk coder against its own "
              r"second pass on " + mac["RerunN"] + r" texts (Visual: a convention was clarified between the passes). "
              r"$^{m}$In the reception models. $^a$Highest level claimed, quadratic-weighted $\kappa$. Runtime locus "
              r"and guardrail codes: Table~\ref{tab:runtimecodes}.}", r"\end{table}"]
    (MS / "tables/tab_supp_codes.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    hv1 = pd.read_csv(RES / "human_round1/human_validation.csv")

    def g1(comp, k):
        r = hv1[(hv1.task == "G") & (hv1.comparison == comp) & (hv1.code == k)]
        return r.kappa.iloc[0] if len(r) else np.nan
    glab = {"live_system": "Runtime, system-initiated", "live_player": "Runtime, player-initiated",
            "gr_constrained": "Constrained scope", "gr_local": "Local or private", "gr_filter": "Filtering",
            "gr_optout": "Player opt-out", "gr_report": "Reporting", "gr_human": "Human monitoring",
            "gr_provider": "Provider policies", "gr_none": "None stated"}
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Human coding of all " + mac["HumGRuntime"] + r" runtime-generation texts, by round}",
             r"\label{tab:runtimecodes}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{Round 1} & \multicolumn{2}{c}{Round 2} \\", r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
             r"Code & $\kappa_{H}$ & $\kappa_{MH}$ & $\kappa_{H}$ & $\kappa_{MH}$ \\", r"\midrule"]
    for k, lab in glab.items():
        r1h = g1(f"{c2} vs {c1}", k); r1m = np.nanmean([g1(f"model vs {c}", k) for c in (c1, c2)])
        lines.append(f"{lab} & {f2(r1h)} & {f2(r1m)} & {f2(hh('G', k))} & {f2(vs_h('G', 'model', k))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\par\smallskip\parbox{\columnwidth}{\scriptsize $\kappa_{H}$: between the two human coders; "
              r"$\kappa_{MH}$: final codes against each coder, averaged. Guardrail codes are scored on texts the "
              r"reference coder marked as runtime generation. Between the rounds, both coders attended a calibration "
              r"session based only on the codebook and constructed examples, then recoded every text (with " +
              mac["HumGDecoy"] + r" decoys).}", r"\end{table}"]
    (MS / "tables/tab_supp_runtime.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    vrows = [("H1_log_reviews_allwindow", "disclosed", M, "All games in the window"),
             ("log_reviews_multi_pooled", "disclosed", R, r"Multi-game developers, pooled"),
             ("log_reviews_developer_fe", "disclosed", R, "Multi-game, developer FE")]
    wv = [(LOC.format("runtime"), "Runtime generation"), (LOC.format("marketing_only"), "Marketing only"),
          (AUT.format("review"), "Review or selection"), (AUT.format("edit"), "Editing"),
          (AUT.format("human_final"), "Human-made final content"), ("ai_on_human", "AI applied to human work"),
          ("d_visual", "Visual"), ("d_voice", "Voice"), ("d_music", "Music"), ("d_sfx", "Sound effects"),
          ("d_text", "Text"), ("d_localization", "Localization"), ("d_code", "Code")]
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{Disclosure and review volume (OLS, log points)}",
             r"\label{tab:volume}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule", r" & $b$ & (SE) & $p$ & $p_{\mathrm{Holm}}$ & $N$ \\", r"\midrule",
             r"\multicolumn{6}{@{}l}{\emph{A. Disclosure (H1)}} \\"]

    def vline(fr, model, t, lab):
        r = fr[(fr.model == model) & (fr.term == t)].iloc[0]
        ph = r.get("p_holm", np.nan)
        bare = lambda q: pv(q).replace("p=", "").replace("p<", "{<}")
        return f"\\quad {lab} & {sgn(r.b)} & ({r.se:.2f}) & {bare(r.p)} & {'--' if pd.isna(ph) else bare(ph)} & {num(r.n)} \\\\"
    for model, t, fr, lab in vrows:
        lines.append(vline(fr, model, t, lab))
    lines.append(r"\multicolumn{6}{@{}l}{\emph{B. Content, disclosing games (H2--H4)}} \\")
    for t, lab in wv:
        lines.append(vline(M, "within_log_reviews", t, lab))
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\par\smallskip\parbox{\columnwidth}{\scriptsize Outcome: log of one plus the number of reviews. "
              r"Developer-clustered standard errors; all covariates. Panel A uses all games released in the window, "
              r"to avoid conditioning on the outcome; panel B uses the disclosing games of the reception sample, with "
              r"the specification of Table~\ref{tab:within}. $p_{\mathrm{Holm}}$: Holm-adjusted within the "
              r"hypothesis family. Panel A: coefficient on disclosure; multi-game: developers with $\geq$2 games; FE: fixed effects.}", r"\end{table}"]
    (MS / "tables/tab_supp_volume.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Established third-party publishers (OLS and linear probability models, percentage points)}",
             r"\label{tab:publisher}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule", r" & $b$ & (SE) & $p$ & $p_{\mathrm{Holm}}$ & $N$ \\", r"\midrule"]

    def pline(model, t, lab, s=100):
        r = P[(P.model == model) & (P.term == t)].iloc[0]
        ph = r.get("p_holm", np.nan)
        bare = lambda q: pv(q).replace("p=", "").replace("p<", "{<}")
        return (f"\\quad {lab} & {sgn(r.b * s)} & ({r.se * s:.2f}) & {bare(r.p)} & "
                f"{'--' if pd.isna(ph) else bare(ph)} & {num(r.n)} \\\\")
    lines.append(r"\multicolumn{6}{@{}l}{\emph{A. Disclosure, all cohort games}} \\")
    for model, t, lab in (("disclosed", "pub_est", "Established publisher"),
                          ("devfe_disclosed", "pub_est", "Same developer (FE)"),
                          ("devfe_nocov_disclosed", "pub_est", "Same developer, no covariates"),
                          ("broad_disclosed", "pub_sep", "Any separate publisher"),
                          ("est5_disclosed", "pub_est5", r"Publisher with $\geq$5 developers"),
                          ("any_disclosed", "pub_est_any", "Including co-published games"),
                          ("other_disclosed", "pub_est_other", "With another developer's game")):
        lines.append(pline(model, t, lab))
    lines.append(r"\multicolumn{6}{@{}l}{\emph{B. Content, disclosing games}} \\")
    for model, lab in (("any_oversight", "Claims human oversight"), ("live_any", "Runtime generation"),
                       ("marketing_only", "Marketing only")):
        lines.append(pline(model, "pub_est", lab))
    lines.append(r"\multicolumn{6}{@{}l}{\emph{C. Reception: disclosure gap}} \\")
    for model, lab, s in (("pct_positive", "Positive reviews", 1), ("crit_share", "AI-critical share", 100)):
        lines.append(pline(model, "disclosed", lab + ", other games", s))
        lines.append(pline(model, "disclosed_if_publisher", lab + ", publisher", s))
        lines.append(pline(model, "disclosed:pub_est", lab + ", difference", s))
    lines.append(r"\multicolumn{6}{@{}l}{\emph{D. Reception: publisher difference, disclosing games}} \\")
    for model, lab, s in (("pct_positive", "Positive reviews", 1), ("crit_share", "AI-critical share", 100)):
        lines.append(pline(model, "publisher_if_disclosed", lab, s))
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\par\smallskip\parbox{\columnwidth}{\scriptsize Established publisher: a listed publisher whose name "
              r"matches none of the game's developers (co-published games count as self-published) and that has released games of at least two developers other than "
              r"itself on Steam, the game's developers included. Covariates of the main models, with this indicator in place of self-publishing. Standard errors clustered by "
              r"developer and by publisher; fixed-effects (FE) rows by developer, among developers with $\geq$2 cohort "
              r"games. Alternative definitions in panel A: any publisher not matching a developer; one with at least five other "
              r"developers; co-published games included; one also listed on another developer's game. "
              r"$p_{\mathrm{Holm}}$: Holm-adjusted across the six RQ3 tests. Panel B: the " + mac["ReceptionCoded"] +
              r" coded disclosures of the reception sample. Panel C: coefficient on disclosure for games without and "
              r"with an established publisher, and their difference. Panel D: games with an established publisher "
              r"minus other games, among disclosing games (the publisher and interaction coefficients summed).}", r"\end{table}"]
    (MS / "tables/tab_supp_publisher.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(len(mac), "macros written")


if __name__ == "__main__":
    main()
