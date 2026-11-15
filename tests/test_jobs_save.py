import json
import time

from vi_web.config import WebConfig
from vi_web.crawl import product_slug
from vi_web.jobs import JobStore


def _run(store, url):
    job = store.start(url, None)
    while job.state in {"waiting_login", "crawling", "analyzing"}:
        time.sleep(0.05)
    return job


def test_crawl_is_saved_with_the_link_name_and_a_recrawl_replaces_it(tmp_path):
    first = [{"comment": "Sản phẩm rất tốt", "rating_star": 5}]
    second = first + [{"comment": "Giao chậm", "rating_star": 2}]
    batches = iter([first, second, second])
    store = JobStore(WebConfig(saved_dir=tmp_path), crawl_fn=lambda *a: next(batches), analyze_fn=lambda i: {"n": len(i)})
    (tmp_path / "crawled").mkdir()
    (tmp_path / "crawled" / "123_456.json").write_text("[]", encoding="utf-8")  # an older save named by ids only

    assert _run(store, "https://shopee.vn/Ban-phim-co-F75-i.123.456").state == "done"
    assert [f.name for f in (tmp_path / "crawled").iterdir()] == ["Ban-phim-co-F75_123_456.json"]

    assert _run(store, "https://shopee.vn/product/123/456").state == "done"  # no name in this link
    files = list((tmp_path / "crawled").iterdir())
    assert [f.name for f in files] == ["123_456.json"]
    assert json.loads(files[0].read_text(encoding="utf-8")) == second


def test_product_slug():
    assert product_slug("https://shopee.vn/B%E1%BA%A3n-ph%C3%ADm-F75-i.1.2?sp=x") == "Bản-phím-F75"
    assert product_slug("https://shopee.vn/product/1/2") == ""


if __name__ == "__main__":  # no pytest in requirements.txt: `python -m tests.test_jobs_save` works too
    import pathlib
    import tempfile

    test_crawl_is_saved_with_the_link_name_and_a_recrawl_replaces_it(pathlib.Path(tempfile.mkdtemp()))
    test_product_slug()
    print("ok")
