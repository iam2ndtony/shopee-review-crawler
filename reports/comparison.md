# Run comparison

| Run | Model | Weight power | UIT-VSFC test Macro-F1 | Shopee Macro-F1 [95% CI] | Neutral precision | Neutral recall | Shopee accuracy | Latency batch 1 (ms) | Latency batch 32 (ms/review) |
|---|---|---|---|---|---|---|---|---|---|
| phobert-p0.25 | vinai/phobert-base | 0.25 | 0.7221 | 0.5525 [0.4828, 0.5868] | 0.0976 | 0.0435 | 0.9143 | 27.5 | 1.95 |
| phobert-p0.5 | vinai/phobert-base | 0.5 | 0.6948 | 0.5643 [0.4852, 0.6045] | 0.0833 | 0.087 | 0.9038 | 26.16 | 2.39 |
| phobert-p1.0 | vinai/phobert-base | 1.0 | 0.7334 | 0.5363 [0.4216, 0.6167] | 0.0773 | 0.1739 | 0.851 | 27.84 | 2.41 |
| xlmr-p0.5 | FacebookAI/xlm-roberta-base | 0.5 | 0.6183 | 0.5209 [0.4541, 0.5421] | 0.0286 | 0.0109 | 0.8983 | 27.3 | 1.21 |

Shopee scores are pooled held-out-product predictions (leave-one-product-out). UIT-VSFC is the shipped (Shopee-trained) model on the 1,600-row test split. Latency is end to end (clean + segment + tokenize + forward) on one GPU; XLM-R runs skip segmentation. All rows were measured back to back in one session; latency still varies by about 2x between sessions (an earlier session measured 13 ms for phobert-p1.0), so compare latency only within this table. phobert-p1.0 was trained before the classifier head's random start was seeded, so retraining it would not reproduce it exactly; its numbers come from its saved predictions. The other runs share one seeded start, so the p1.0 versus p0.5/p0.25 difference mixes the weight power with that change.

Notes:
- Held-out sets are not a human-labeled Gold Set; no kappa exists.
- Shopee Neutral evaluation rows: 92; scores are rough.
- Shopee interval resamples whole products (5), so it is wide and itself rough; UIT-VSFC intervals resample rows.
- Shopee labels are weak (star ratings); star-versus-text mismatches are not corrected.
- UIT-VSFC uses this project's own 80/10/10 split, not comparable with published results.
- A product with no Neutral rows is scored on 2 classes; low_confidence is computed on pooled data only.
- Latency is per review on this machine's GPU, batch 1 and 32; end_to_end = clean + segment + tokenize + forward.
