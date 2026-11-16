import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.textpath import TextPath
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import ScaledTranslation, blended_transform_factory

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "manuscript/figures"
FIG.mkdir(parents=True, exist_ok=True)

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
AUTH_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
SIG, NS, BAND = AUTH_RAMP[2], "#9b9a97", "#f1f0ec"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.it": "sans:italic", "mathtext.rm": "sans", "mathtext.cal": "sans",
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "pdf.fonttype": 42,
})


def prevalence():
    df = pd.read_csv(ROOT / "data/processed/games.csv", usecols=["release_date", "has_page", "disclosed",
                                                                 "review_count"], low_memory=False)
    df = df[(df.has_page == 1) & (df.release_date >= "2024-01-01")]
    df["m"] = pd.to_datetime(df.release_date.str[:7])
    allg = df.groupby("m").disclosed.mean() * 100
    rv = df[(df.review_count >= 10) & (df.release_date <= "2026-06-30")].groupby("m").disclosed.mean() * 100
    fig, ax = plt.subplots(figsize=(3.5, 1.7))
    ax.plot(allg.index, allg.values, color=SIG, lw=1.2, label="All releases", solid_joinstyle="round")
    ax.plot(rv.index, rv.values, color=ORANGE, lw=1.2, label="Releases with \u226510 reviews", solid_joinstyle="round")
    ax.axvline(pd.Timestamp("2026-01-01"), color=INK2, lw=0.5, ls=(0, (2.5, 2)))
    ax.set_ylim(0, 50)
    ax.text(pd.Timestamp("2026-01-20"), 49, "Valve rule\nrevision", fontsize=6, color=INK2, va="top", linespacing=1.1)
    for ser in (allg, rv):
        ax.annotate(f"{ser.values[-1]:.0f}%", xy=(ser.index[-1], ser.values[-1]), xytext=(3, 0),
                    textcoords="offset points", fontsize=6, color=INK, va="center")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.set_xlim(pd.Timestamp("2023-12-01"), pd.Timestamp("2026-12-15"))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.tick_params(which="major", length=2.5, width=0.6, pad=2)
    ax.tick_params(which="minor", length=1.5, width=0.5)
    ax.set_ylabel("Share with AI disclosure (%)")
    ax.legend(frameon=False, loc="upper left", handlelength=1.6, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "fig_prevalence.pdf")
    plt.close(fig)


def composition():
    ct = pd.read_csv(RES / "locus_by_authority.csv", index_col=0)
    ns = pd.read_csv(RES / "locus_counts.csv", index_col=0)["n"]
    order = ["dev_time", "runtime", "marketing_only"]
    names = {"dev_time": "Development time", "runtime": "Runtime", "marketing_only": "Marketing only"}
    cols = ["none", "review", "edit", "human_final"]
    lab = {"none": "None stated", "review": "Review", "edit": "Edit", "human_final": "Human final"}
    fig, ax = plt.subplots(figsize=(3.5, 1.75))
    for i, loc in enumerate(order):
        left = 0
        for j, c in enumerate(cols):
            v = ct.loc[loc, c] * 100
            ax.barh(i, v, left=left, height=0.6, color=AUTH_RAMP[j], edgecolor="white", linewidth=0.8,
                    label=lab[c] if i == 0 else None)
            if v >= 4:
                ax.text(left + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=6,
                        color=INK if j == 0 else "white")
            left += v
    ax.set_yticks(range(3), [f"{names[l]}\n$n$ = {ns[l]:,}" for l in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.set_xlabel("Share of disclosures (%)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=3, labelcolor=INK)
    ax.tick_params(axis="x", length=2.5, width=0.6, pad=2)
    leg = ax.legend(ncol=4, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1.0,
                    handleheight=0.8, columnspacing=1.0, handletextpad=0.4, borderaxespad=0.3,
                    title="Highest stated human authority", title_fontsize=7)
    leg._legend_box.align = "left"
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "fig_composition.pdf")
    plt.close(fig)


def _tw(text, size, weight="normal", style="normal"):
    fp = FontProperties(family="sans-serif", weight=weight, style=style)
    return TextPath((0, 0), text, size=size, prop=fp).get_extents().width / 72


def _chip(ax, x, y, text, fs=6.3, fill="white", edge=INK2, color=INK, dashed=False, h=0.165, pad=0.045):
    w = _tw(text, fs) + 2 * pad
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.035", fc=fill,
                                ec=edge, lw=0.6, ls=(0, (2, 1.2)) if dashed else "-"))
    ax.text(x + w / 2, y - 0.004, text, ha="center", va="center", fontsize=fs, color=color)
    return w


def overview():
    H = json.load(open(RES / "headline.json"))
    M = pd.read_csv(RES / "models.csv").set_index(["model", "term"])
    R = pd.read_csv(RES / "robustness.csv").set_index(["model", "term"])
    n_rev = H["n_reviews_collected"]
    loc, aut = "C(locus, Treatment('dev_time'))[T.{}]", "C(auth, Treatment('none'))[T.{}]"

    def b(model, term, sc=1, src=M):
        return f"{src.loc[(model, term), 'b'] * sc:+.1f}".replace("-", "−")

    W, Hh = 7.16, 2.50
    fig = plt.figure(figsize=(W, Hh))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, Hh)
    ax.axis("off")
    xa, xb, xc, xe = 0.04, 1.68, 5.02, 7.12
    gap = 0.12
    rows = [2.07, 1.67, 1.25, 0.85, 0.46]
    seps = [1.87, 1.46, 1.04, 0.66, 0.26]
    tint, sep_c = "#edf3fb", "#d3dbe6"

    for x0, x1, letter, t in ((xa, xb - gap, "a", "Steam census"), (xb, xc - gap, "b", "Authority-allocation coding"),
                              (xc, xe, "c", "Player reception")):
        ax.text(x0, 2.39, letter, fontsize=8, fontweight="bold", color=INK, va="center")
        ax.text(x0 + 0.14, 2.39, t, fontsize=7.5, fontweight="bold", color=INK, va="center")
        ax.plot([x0, x1], [2.27, 2.27], color=INK2, lw=0.6)

    ax.add_patch(Rectangle((xb, seps[-1]), xc - gap - xb, seps[0] - seps[-1], fc=tint, ec="none", zorder=0))
    for yy in seps[:-1]:
        ax.plot([xb, xe], [yy, yy], color=sep_c, lw=0.5, zorder=1)
    ax.plot([xb, xe], [seps[-1], seps[-1]], color=INK2, lw=0.6)
    ax.plot([xa, xb - gap], [seps[-1], seps[-1]], color=INK2, lw=0.6)

    ax.add_patch(Rectangle((0.10, 1.03), 1.36, 1.15, fc="white", ec=INK2, lw=0.6))
    ax.add_patch(Rectangle((0.10, 2.08), 1.36, 0.10, fc=INK2, ec=INK2, lw=0.6))
    ax.add_patch(Rectangle((0.16, 1.72), 0.52, 0.30, fc=GRID, ec="none"))
    for k, yy in enumerate((1.97, 1.89, 1.81, 1.75)):
        ax.plot([0.74, 1.40 - 0.12 * (k % 2)], [yy, yy], color=GRID, lw=2.2, solid_capstyle="butt")
    ax.add_patch(Rectangle((0.14, 1.09), 1.28, 0.55, fc="white", ec=BLUE, lw=0.9))
    ax.text(0.78, 1.555, "AI Generated Content Disclosure", fontsize=5.6, fontweight="bold", color=INK, va="center",
            ha="center")
    ax.text(0.20, 1.33, "“… images were generated\nand human reviewed”", fontsize=6.3, style="italic",
            color=INK, va="center", linespacing=1.15)
    stats = [f"{H['n_cohort_with_page']:,} games, Jan. 2024–Oct. 2026",
             f"{H['n_disclosed']:,} disclosures ({100 * H['share_disclosed']:.1f}%)",
             f"{n_rev:,} reviews collected"]
    for k, t in enumerate(stats):
        assert 0.10 + _tw(t, 6.5) <= xb - gap, f"panel A overflow: {t}"
        ax.text(0.10, 0.83 - 0.20 * k, t, fontsize=6.5, color=INK, va="center")
    ax.annotate("", xy=(xb + 0.02, 1.36), xytext=(1.42, 1.36),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8, mutation_scale=7, shrinkA=0, shrinkB=0))

    xt, xchip = xb + 0.07, xb + 0.72
    titles = ["Disclosure", "Locus", "Human role", "Domain", "Guardrails"]
    for t, yy in zip(titles, rows):
        ax.text(xt, yy, t, fontsize=7, fontweight="bold", color=INK, va="center")
    sub = dict(fontsize=5.8, style="italic", color=INK2, ha="center", va="center")

    w = _chip(ax, xchip, rows[0], "Present / absent", fill=GRID, edge=GRID)
    ax.text(xchip + w + 0.08, rows[0], "binary label in prior work", fontsize=6, style="italic", color=INK2,
            va="center")

    def slots(y, items, gap_):
        x = xchip
        for item in items:
            if item is None:
                x += 0.05
                continue
            lab, note, kw = item
            cw, nw = _tw(lab, 6.3) + 0.09, _tw(note, 5.8, style="italic")
            sw = max(cw, nw)
            _chip(ax, x + (sw - cw) / 2, y + 0.05, lab, **kw)
            ax.text(x + sw / 2, y - 0.10, note, **sub)
            x += sw + gap_
        return x - gap_

    right = xc - gap - 0.06
    x_end = slots(rows[1], [("Development time", "developer in the loop", {}), ("Runtime", "developer on the loop", {}),
                            ("Marketing only", "outside play", {})], 0.10)
    ramp = [dict(fill=c, edge=c, color=INK if j == 0 else "white") for j, c in enumerate(AUTH_RAMP)]
    x_end = max(x_end, slots(rows[2], [("None", "unstated", ramp[0]), ("Review", "veto", ramp[1]),
                                       ("Edit", "modify", ramp[2]), ("Human final", "author", ramp[3]), None,
                                       ("AI on human work", "reverse delegation", dict(edge=AUTH_RAMP[2]))], 0.025))
    assert x_end <= right, f"panel B overflow: {x_end:.2f} > {right:.2f}"

    x = xchip
    for lab in ("Visual", "Voice", "Music", "Text", "Localization", "Code"):
        x += _chip(ax, x, rows[3], lab) + 0.05

    for line, labs in ((0.09, ("Scope limits", "Local or private", "Filtering", "Opt-out")),
                       (-0.10, ("Reporting", "Monitoring", "Provider", "None stated"))):
        x = xchip
        for lab in labs:
            x += _chip(ax, x, rows[4] + line, lab, dashed=lab == "None stated") + 0.05

    for yy in rows:
        ax.annotate("", xy=(xc - 0.03, yy), xytext=(xc - gap - 0.02, yy),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.6, mutation_scale=6, shrinkA=0, shrinkB=0))

    fe = b("developer_fe", "disclosed", src=R)
    fec = b("crit_share_developer_fe", "disclosed", 100, src=R)
    finds = [("H1", [f"Rating {b('H1_pct_positive', 'disclosed')} pp; within developer {fe} n.s.",
                     f"AI-critical share {fec} pp within developer"]),
             ("H2", [f"Marketing only {b('within_crit_share', loc.format('marketing_only'), 100)} pp AI-critical",
                     f"Runtime {b('within_crit_share', loc.format('runtime'), 100)} pp AI-critical, flag-dependent"]),
             ("H3", ["Review, edit, human final: n.s.",
                     f"AI on human work {b('within_pct_positive', 'ai_on_human')} pp rating"]),
             ("H4", [f"Visual {b('within_crit_share', 'd_visual', 100)} pp AI-critical; rating n.s.",
                     "Voice: n.s."]),
             ("RQ2", [f"No guardrail stated: {100 * H['live_in_gr_none']:.0f}% of fully",
                      "coded runtime disclosures"])]
    for (tag, lines), yy in zip(finds, rows):
        assert all(xc + 0.36 + _tw(t, 6.5) <= xe for t in lines), f"panel C overflow: {lines}"
        ax.text(xc + 0.04, yy, tag, fontsize=7, fontweight="bold", color=INK, va="center")
        ax.text(xc + 0.36, yy, "\n".join(lines), fontsize=6.5, color=INK, va="center", linespacing=1.3)

    note = dict(fontsize=6, style="italic", color=INK2, va="center")
    ax.text(xa + 0.06, 0.12, "Snapshot of 24 Oct. 2026", **note)
    ax.text(xb + 0.07, 0.12, "LLM coder checked against reference labels and two human coders", **note)
    ax.text(xc + 0.04, 0.12, "OLS with covariates; developer FE; Holm", **note)
    fig.savefig(FIG / "fig_overview.pdf")
    plt.close(fig)


TERMS = [("C(locus, Treatment('dev_time'))[T.runtime]", "Runtime generation", "Locus (vs. development time)"),
         ("C(locus, Treatment('dev_time'))[T.marketing_only]", "Marketing only", None),
         ("C(auth, Treatment('none'))[T.review]", "Review", "Authority (vs. none stated)"),
         ("C(auth, Treatment('none'))[T.edit]", "Edit", None),
         ("C(auth, Treatment('none'))[T.human_final]", "Human final", None),
         ("ai_on_human", "AI applied to human work", None),
         ("d_visual", "Visual", "Domain"), ("d_voice", "Voice", None), ("d_music", "Music", None),
         ("d_sfx", "Sound effects", None), ("d_text", "Text", None), ("d_localization", "Localization", None),
         ("d_code", "Code", None)]


def _signed(v):
    return f"{v:+.1f}".replace("-", "\u2212")


def coef_figure():
    m = pd.read_csv(RES / "models.csv")
    panels = [("within_pct_positive", "Positive reviews (percentage points)", 1, (-10.8, 8.4)),
              ("within_crit_share", "AI-critical reviews (percentage points)", 100, (-6.4, 10.4))]
    panels = [p for p in panels if (m.model == p[0]).any()]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.16, 2.10), sharey=True)
    axes = np.atleast_1d(axes)
    ypos, y = [], 0
    for t, _, grp in TERMS:
        if grp and ypos:
            y += 1
        ypos.append(y)
        y += 1
    starts = [i for i, (_, _, grp) in enumerate(TERMS) if grp] + [len(TERMS)]
    bands = [(ypos[a] - 1.43, ypos[b - 1] + 0.57) for a, b in zip(starts[:-1], starts[1:])][1::2]
    for ax, (model, title, scale, xlim), letter in zip(axes, panels, "ab"):
        mm = m[m.model == model].set_index("term")
        ax.set_facecolor("none")
        ax.axvline(0, color=INK2, lw=0.5, ls=(0, (2.5, 2)), zorder=1)
        for (t, lab, grp), yy in zip(TERMS, ypos):
            if t not in mm.index:
                continue
            r = mm.loc[t]
            sig = r.p_holm < 0.05 if not np.isnan(r.get("p_holm", np.nan)) else r.p < 0.05
            b, lo, hi = r.b * scale, r.lo * scale, r.hi * scale
            c = SIG if sig else NS
            ax.plot([lo, hi], [yy, yy], color=c, lw=1.3 if sig else 0.9, solid_capstyle="round", zorder=2)
            ax.plot(b, yy, "o", ms=4.6 if sig else 4.0, mfc=c if sig else "white", mec="white" if sig else c,
                    mew=0.7 if sig else 0.9, zorder=3)
            if sig:
                out = hi if b > 0 else lo
                ax.annotate(_signed(b), xy=(out, yy), xytext=(3 if b > 0 else -3, 0), textcoords="offset points",
                            ha="left" if b > 0 else "right", va="center", fontsize=6, color=INK)
        n = int(mm.n.iloc[0])
        ax.text(1, 0.015, f"$n$ = {n:,} games", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6, color=INK2)
        ax.set_xlim(*xlim)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.set_xlabel(title)
        ax.text(0, 1, letter, transform=ax.transAxes + ScaledTranslation(-9 / 72, 5 / 72, fig.dpi_scale_trans),
                ha="left", va="baseline", fontsize=8, fontweight="bold", color=INK)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(INK2)
        ax.tick_params(axis="y", length=0, pad=3, colors=INK, labelcolor=INK)
        ax.tick_params(axis="x", length=2.5, width=0.6, pad=2)
    axes[0].set_yticks(ypos, [lab for _, lab, _ in TERMS])
    axes[0].set_ylim(ypos[-1] + 0.75, -1.45)
    for (t, lab, grp), yy in zip(TERMS, ypos):
        if grp:
            axes[0].annotate(grp, xy=(0, yy - 0.85), xycoords=("axes fraction", "data"), xytext=(-3, 0),
                             textcoords="offset points", ha="right", va="center", fontsize=7, color=INK,
                             fontweight="bold", annotation_clip=False)
    handles = [Line2D([], [], color=SIG, lw=1.3, marker="o", ms=4.6, mfc=SIG, mec="white", mew=0.7),
               Line2D([], [], color=NS, lw=0.9, marker="o", ms=4.0, mfc="white", mec=NS, mew=0.9)]
    axes[-1].legend(handles, ["Holm-adjusted $p$ < .05", "Holm-adjusted $p$ ≥ .05"], loc="lower right",
                    bbox_to_anchor=(1, 1), ncol=2, frameon=False, handlelength=1.6, handletextpad=0.4,
                    columnspacing=1.2, borderaxespad=0.15, borderpad=0)
    fig.tight_layout(pad=0.3, w_pad=2.0)
    tr = blended_transform_factory(fig.transFigure, axes[0].transData)
    x1 = axes[-1].get_position().x1
    for top, bottom in bands:
        fig.patches.append(Rectangle((0.004, top), x1 - 0.004, bottom - top, transform=tr, facecolor=BAND,
                                     lw=0, zorder=-1))
    return fig


def coef():
    fig = coef_figure()
    fig.savefig(FIG / "fig_coef.pdf")
    plt.close(fig)


if __name__ == "__main__":
    overview()
    prevalence()
    composition()
    coef()
    print("figures written to", FIG)
