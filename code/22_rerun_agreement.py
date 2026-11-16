import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "annotation/bulk"
CHANGED = {"d_visual"}


def load(p):
    with open(p, encoding="utf-8") as f:
        return {r["uid"]: r for r in map(json.loads, f) if r}


def main():
    a, b = load(B / "coded_batch_02_v1.jsonl"), load(B / "coded_batch_02.jsonl")
    ids = sorted(set(a) & set(b))
    codes = [k for k in a[ids[0]] if k != "uid"]
    rows = []
    for k in codes:
        x = np.array([int(a[i][k]) for i in ids]); y = np.array([int(b[i][k]) for i in ids])
        both = set(x) | set(y)
        kap = cohen_kappa_score(x, y, weights="quadratic" if k == "authority" else None) if len(both) > 1 else np.nan
        rows.append({"code": k, "n": len(ids), "n_pos_v1": int((x > 0).sum()), "n_pos_v2": int((y > 0).sum()),
                     "agree": float((x == y).mean()), "kappa": kap, "convention_changed": k in CHANGED})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/rerun_agreement.csv", index=False)
    s = out[~out.convention_changed & (out.code != "authority") & (out[["n_pos_v1", "n_pos_v2"]].min(axis=1) >= 5)]
    print(out.round(3).to_string(index=False))
    print(f"\n{len(ids)} texts; codes with >=5 positives in both passes (excluding d_visual): {len(s)}; "
          f"median kappa {s.kappa.median():.3f}, min {s.kappa.min():.3f} ({s.loc[s.kappa.idxmin(), 'code']}); "
          f"authority weighted kappa {out.set_index('code').loc['authority', 'kappa']:.3f}; "
          f"max disagreement {1 - out[~out.convention_changed].agree.min():.3f}")


if __name__ == "__main__":
    main()
