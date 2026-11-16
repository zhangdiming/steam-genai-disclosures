import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    cohort = {int(x) for x in (ROOT / "data/processed/cohort_appids.txt").read_text().split()}
    rows = []
    with open(ROOT / "data/raw/store_items.jsonl", encoding="utf-8") as f:
        for line in f:
            try:
                it = json.loads(line)
            except json.JSONDecodeError:
                continue
            if it.get("missing") or it["appid"] not in cohort:
                continue
            names = [d["name"].strip().lower() for d in it.get("basic_info", {}).get("developers", []) or []]
            rows.append({"appid": it["appid"], "developers": json.dumps([n for n in names if n], ensure_ascii=False)})
    out = pd.DataFrame(rows).drop_duplicates("appid", keep="last").sort_values("appid")
    out.to_csv(ROOT / "data/processed/developer_lists.csv", index=False)
    print(len(out), "games;", int(out.developers.map(lambda s: len(json.loads(s)) >= 2).sum()), "with two or more developers")


if __name__ == "__main__":
    main()
