# Web Demo: Shopee Product Review Analysis: Design

Status: draft for user review. Builds on PR #3 (`phobert-experiments`). Branch for the work: `web-demo`, stacked on it (PR #4).

## Purpose
Deliverable 5 of the project brief ("interactive web demo for live inference"), shaped by the user: paste a Shopee product link, get an exploratory analysis (EDA) of that product's reviews using the model's predicted sentiment, and open each review individually. Audience: the project supervisor and the author, running it on the author's own machine.

## Non-goals
Model switcher or side-by-side models, user accounts or sign-in, deployment, history or saving crawled reviews, multiple simultaneous users, word clouds or topic analysis, captcha solving or any way around a Shopee refusal, a framework or build step, external chart libraries, test files (assumed to carry over; confirm), showing the model's name anywhere on the page (user's request).

## Two ways in
- **Live link.** `POST /analyze/link {url, max_reviews?}`: the URL must be `https://shopee.vn/...` (or `www.shopee.vn`) with a shop id and item id (`-i.<shop>.<item>` or `/product/<shop>/<item>`); anything else is HTTP 400. The server starts one background job (HTTP 409 if one is already active). A Chrome window opens with the saved Shopee login profile; the page tells the user to log in there if asked and click "I'm ready" (`POST /analyze/jobs/{id}/ready`), which replaces the terminal ENTER of the root crawler. The job then crawls, cleans, predicts with the final model and builds the analysis. States: `waiting_login`, `crawling`, `analyzing`, `done`, `error`, with review counts as progress. A job that is not marked ready within 10 minutes ends as an error.
- **Saved product.** `GET /analyze/saved` lists the review files in the saved folder (default the repository's `raw_reviews/`, env `SAVED_REVIEWS_DIR`); `POST /analyze/saved {file}` takes a name from that listing only (never a path) and returns the analysis directly. Predictions are the **held-out** ones from the leave-one-product-out run (`data/train/preds/<run>/shopee_oof.csv`), so the model did not see that product; the page says so. Those files store only the label, so saved mode has no probabilities and the detail view hides the probability bars. A saved review matches its prediction by the MD5 key of its cleaned text within the product (the key the pipeline's deduplication uses), so reviews removed as duplicates inherit the kept twin's label. A review with no match counts as "not predicted" and is reported in the summary.
- **Training-data warning.** A live link whose product id equals one of the saved products is flagged: the final model was trained on it, so the predictions are not a fair test; the page suggests the saved route.

## Analysis (per product)
Input rows keep only: review text, star rating, date, variation (`product_items[0].model_name`), and a per-product review number. Reviewer username, user id and avatar are dropped at the crawl boundary and never reach the analysis, the API or the page.
- **Cleaning and flags.** Text goes through the existing `clean_texts` with `lowercase=False`. Stars-only reviews (empty text) are counted but not listed or predicted. Flags: `duplicate` (an earlier review of the same product has the same MD5 key), `repetitive` (`low_ttr_mask`, the pipeline's TTR rule).
- **Coin phrase.** Before cleaning, `strip_coin_phrase` cuts the "nhan xu" phrase ("nhan"/"lay", up to two words, "xu"; matched on text folded to ASCII, so typos and missing tone marks still hit) out of each review and keeps the rest. A short review that is almost all the known phrase is removed whole. `summary` reports `coin_farming_dropped` (removed) and `coin_phrase_trimmed` (cut). In saved mode the held-out prediction is looked up with the review as it was in the dataset, before the cut. Known limit: up to two words between "nhan" and "xu" are cut with the phrase. Applies to the demo only; the training data is unchanged.
- **Prediction (live).** The final model of the served run in batches of 32. The served run is env `VI_RUN`, default `phobert-p0.5` (the run chosen on validation); this default lives in the predictor module and does not change `TrainConfig`.
- **Result JSON.** `summary` (reviews found, with text, stars-only, not predicted, predicted share per class, duplicate count, repetitive count, disagreement count, date range, `mode`: `live` or `saved`, `training_data_warning`: bool), `stars_all` (star counts over all reviews), `sentiment_counts`, `stars_by_label` (5 by 3 matrix over predicted reviews), `by_month` (month, then negative, neutral, positive counts), and `reviews` (number, stars, date, variation, text, cleaned, label, probs or null, flags). "Disagrees with stars" means 4 or 5 stars predicted Negative, or 1 or 2 stars predicted Positive; 3 stars never disagree.
- **Size and limits.** A live crawl keeps only reviews with text (stars-only ones are dropped while crawling, so a live result has no stars-only reviews) and takes every such review Shopee returns (changed from the first design's 1,000 default and 3,000 cap at the user's request); `max_reviews` stays an optional API parameter, and the crawl also stops if the endpoint repeats a page. Jobs live in memory only (at most 5 kept), and nothing is written to disk.

## The page
Static HTML, CSS and JS served by FastAPI at `/` (`GET /` and `/static/*`); no framework, no build step, no request to any other site. English interface. Charts are hand-drawn SVG following the dataviz skill (colour-blind-safe sentiment colours, light and dark, labelled values, a text equivalent for each chart).
- **Top bar:** link box, "Analyze reviews" button, saved-product dropdown, status line (`aria-live`) with the login step and progress.
- **Summary figures:** reviews found, with text and stars-only (separately), predicted Positive, Negative and Neutral shares, repetitive and duplicate reviews, disagreements.
- **Charts:** predicted sentiment counts; star rating by predicted sentiment with disagreement cells highlighted; reviews per month stacked by predicted sentiment.
- **Review browser:** table of stars, predicted label chip, text snippet and flags; filters for predicted label, stars, flagged and "disagrees with stars"; search; 25 rows per page; selecting a row opens the full review (original text, cleaned text, probabilities when present, stars, date, variation, flags). All review text is inserted with `textContent`, never `innerHTML`.
- **Plain caution, one line:** predictions come from a model that is weakest on Neutral, and labels are not human-verified. No model name, no reviewer identity.
- **Quality bar:** keyboard operable, visible focus, labelled controls, usable from 390 px to 1440 px, no sideways scroll, text contrast at WCAG AA.

## Code layout
- New package `vi_web/`: `config.py` (paths, caps, env), `crawl.py` (URL parsing, Playwright crawl on a worker thread, PII drop), `analyze.py` (rows, flags, analysis, saved-product join), `jobs.py` (in-memory job store and states), `routes.py` (the endpoints above), `static/` (`index.html`, `app.js`, `style.css`).
- `vi_clean/predictor.py` (extracted from `vi_clean/api.py`, behaviour unchanged): the lazy classifier loader with its lock and the batch prediction used by `/predict`, `/predict/batch` and the web analysis; default run `phobert-p0.5`.
- `vi_clean/api.py` imports the predictor and mounts the `vi_web` router and static files. `requirements.txt` already lists `playwright`.
- The user's root `shopee_crawler_persistent.py` is not modified or imported (the user has uncommitted edits to it and it blocks on `input()`); `vi_web/crawl.py` reimplements the small fetch loop with the same profile folder (env `SHOPEE_PROFILE_DIR`, default `<repo>/shopee_chrome_profile`), the same ratings endpoint, 50 reviews per request, one second between requests and 3 retries.

## Security and privacy
- The server is for local single-user use: run `uvicorn vi_clean.api:app` on the default `127.0.0.1`; the README says not to expose it (no authentication, and `/jobs/merge-clean` already has none).
- URL host must be exactly `shopee.vn` or `www.shopee.vn`, scheme https, ids digits only; the crawler visits and calls only shopee.vn. JSON-only POSTs mean a foreign web page cannot start a crawl through the browser (a cross-origin JSON POST needs a preflight that the server does not grant).
- No caller-supplied file path anywhere. Reviews exist only in server memory for the life of the job store and in the user's browser tab.

## Errors (each shown as a plain message on the page)
Not a Shopee product link (400); another analysis running (409); Chrome profile already in use (close the other Chrome or crawler); Shopee refused or asked for login or captcha (HTTP 403, 401 or an API error code: log in in the window and try again; no loops, no captcha solving); Playwright or Chrome not installed; no reviews with text found (shown as a result, not an error); model checkpoint missing (503); job not marked ready in time.

## Known limits (reported, not hidden)
- Live crawling can fail at any time because of Shopee's login, captcha and rate rules, and it is against Shopee's terms; the saved route is the dependable demo path.
- Chrome allows one session per profile: the author's own crawler and the web crawl cannot run together.
- The pipeline deduplicated all products together, so a short generic review that also appears in another product (or in UIT-VSFC) has no held-out prediction for this product. It counts as "not predicted"; borrowing the other product's label would leak, because that product's fold model trained on this one.
- Saved-mode labels have no probabilities. Neutral is the weakest class (Shopee Neutral recall under 0.2 in the measured runs); about half of Shopee reviews are stars-only and cannot be predicted.
- The model was trained on UIT-VSFC and 5 Shopee products; predictions on other categories may be worse (per-product Macro-F1 ranged 0.37 to 0.80).

## Success criteria
- `GET /` serves the page; `GET /analyze/saved` lists the 5 saved products.
- For each saved product, the analysis numbers cross-check against the source: reviews found equals the raw count, with-text plus stars-only equals found, predicted counts plus not-predicted equal with-text, and every row of that product in `shopee_oof.csv` is matched by at least one review and carries the same label.
- A live crawl of one real product link works end to end on the author's machine with the author's login, and a deliberate bad link, a busy server and a Shopee refusal each produce the plain message above.
- The page works at 1440, 1024, 768 and 390 px with keyboard only, and shows no reviewer username or id anywhere, in the API response or in the page.
