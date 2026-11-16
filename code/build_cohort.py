import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START, END = datetime(2024, 1, 1), datetime(2026, 10, 24)


def parse_date(s):
    if not s:
        return None
    for fmt in ("%b %d, %Y", "%d %b, %Y", "%B %d, %Y", "%d %B, %Y"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            pass
    return None


def main():
    seen, keep = set(), []
    with (ROOT / "data/raw/search_index.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["appid"] in seen:
                continue
            seen.add(r["appid"])
            d = parse_date(r["released_text"])
            if d and START <= d <= END:
                keep.append(r["appid"])
    out = ROOT / "data/processed/cohort_appids.txt"
    out.write_text("\n".join(map(str, keep)) + "\n")
    print(len(seen), "unique listings;", len(keep), "in cohort")


if __name__ == "__main__":
    main()
