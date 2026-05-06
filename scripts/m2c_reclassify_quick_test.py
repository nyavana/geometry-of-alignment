"""Re-classify quick_test/evaluation_results.json under the fixed
`classify_refusal` regex (the original missed "I cannot ___" responses
that the abliterated model emits in place of the base "I'm sorry, but I
can't ___" form). Writes evaluation_results_reclassified.{json,csv}
without overwriting the original eval, so the diff is auditable.
"""

import csv
import json
from pathlib import Path

from src.benchmark.classify_refusal import classify_refusal


def main():
    base = Path("/home/nyavana/columbia/6699/shared/results/agent/abliteration/refusal_rates/quick_test")
    src = base / "evaluation_results.json"
    with open(src) as f:
        rows = json.load(f)
    out_rows = []
    for r in rows:
        new_is_refusal = classify_refusal(r["response"])
        new_actual = "refuse" if new_is_refusal else "comply"
        new_correct = (r["expected"] == "refuse") == new_is_refusal
        out_rows.append({
            **r,
            "actual": new_actual,
            "correct": new_correct,
            "over_refusal": r["expected"] == "comply" and new_is_refusal,
            "under_refusal": r["expected"] == "refuse" and not new_is_refusal,
            "actual_old": r["actual"],
            "flipped": new_actual != r["actual"],
        })
    out_json = base / "evaluation_results_reclassified.json"
    out_csv = base / "evaluation_results_reclassified.csv"
    with open(out_json, "w") as f:
        json.dump(out_rows, f, indent=2)
    fieldnames = ["prompt_id", "variant_idx", "category", "expected", "actual",
                  "actual_old", "flipped", "correct", "over_refusal",
                  "under_refusal", "prompt", "response"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in out_rows:
            w.writerow(row)
    flipped = sum(1 for r in out_rows if r["flipped"])
    refused = sum(1 for r in out_rows if r["actual"] == "refuse")
    sr_refused = sum(1 for r in out_rows
                     if r["category"] == "should_refuse" and r["actual"] == "refuse")
    sr_total = sum(1 for r in out_rows if r["category"] == "should_refuse")
    print(f"Wrote {out_json}")
    print(f"  Flipped: {flipped}/{len(out_rows)}")
    print(f"  Refused (after fix): {refused}/{len(out_rows)} "
          f"({100*refused/len(out_rows):.1f}%)")
    print(f"  should_refuse refusal rate: {sr_refused}/{sr_total} "
          f"({100*sr_refused/sr_total:.1f}%)")


if __name__ == "__main__":
    main()
