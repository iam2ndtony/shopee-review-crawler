import json
import time

from vi_web.config import WebConfig
from vi_web.jobs import JobStore


def _run(store, url):
    job = store.start(url, None)
    while job.state in {"waiting_login", "crawling", "analyzing"}:
        time.sleep(0.05)
    return job


def test_crawl_is_saved_under_crawled_and_recrawl_replaces_it(tmp_path):
    first = [{"comment": "Sản phẩm rất tốt", "rating_star": 5}]
    second = first + [{"comment": "Giao chậm", "rating_star": 2}]
    batches = iter([first, second])
    store = JobStore(WebConfig(saved_dir=tmp_path), crawl_fn=lambda *a: next(batches), analyze_fn=lambda i: {"n": len(i)})

    assert _run(store, "https://shopee.vn/x-i.123.456").state == "done"
    assert _run(store, "https://shopee.vn/x-i.123.456").state == "done"

    files = list((tmp_path / "crawled").iterdir())
    assert [f.name for f in files] == ["123_456.json"]
    assert json.loads(files[0].read_text(encoding="utf-8")) == second


if __name__ == "__main__":  # no pytest in requirements.txt: `python -m tests.test_jobs_save` works too
    import pathlib
    import tempfile

    test_crawl_is_saved_under_crawled_and_recrawl_replaces_it(pathlib.Path(tempfile.mkdtemp()))
    print("ok")
