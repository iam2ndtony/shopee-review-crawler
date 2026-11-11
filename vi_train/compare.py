"""Build reports/comparison.md from every reports/<run>.json (no number is typed by hand).

Run: python -m vi_train.compare
One row per run. Accuracy is the last metric column on purpose (it misleads under imbalance).
"""
import json

from .config import TrainConfig


def main() -> None:
    cfg = TrainConfig()
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(cfg.reports_dir.glob("*.json"))]
    head = ("| Run | Model | Weight power | UIT-VSFC test Macro-F1 | Shopee Macro-F1 [95% CI] | "
            "Neutral precision | Neutral recall | Shopee accuracy | Latency batch 1 (ms) | Latency batch 32 (ms/review) |")
    lines = [head, "|" + "---|" * 10]
    for r in reports:
        s, n = r["shopee_oof"], r["shopee_oof"]["per_class"]["neutral"]
        lat = r["latency"]
        lines.append(
            f"| {r['run']['name']} | {r['run']['model_name']} | {r['run']['weight_power']} "
            f"| {r['uit_vsfc_test_final']['macro_f1']} | {s['macro_f1']} [{s['macro_f1_ci'][0]}, {s['macro_f1_ci'][1]}] "
            f"| {n['precision']} | {n['recall']} | {s['accuracy']} "
            f"| {lat['batch1']['end_to_end']['median_ms']} | {lat['batch32']['end_to_end']['median_ms']} |")
    notes = reports[0]["notes"] if reports else []
    body = ["# Run comparison", "", *lines, "", "Shopee scores are pooled held-out-product predictions "
            "(leave-one-product-out). UIT-VSFC is the shipped (Shopee-trained) model on the 1,600-row test split. "
            "Latency is end to end (clean + segment + tokenize + forward) on one GPU; XLM-R runs skip segmentation.",
            "", "Notes:", *[f"- {x}" for x in notes if not x.startswith("Forgetting")], ""]
    out = cfg.reports_dir / "comparison.md"
    out.write_text("\n".join(body), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
