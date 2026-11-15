import json

from vi_web.analyze import list_saved
from vi_web.config import WebConfig


def test_crawled_products_are_listed_after_saved_ones(tmp_path):
    (tmp_path / "crawled").mkdir()
    (tmp_path / "a.json").write_text(json.dumps([{"comment": "tốt"}]), encoding="utf-8")
    (tmp_path / "crawled" / "1_2.json").write_text(json.dumps([{"comment": "ok"}, {"comment": "kém"}]), encoding="utf-8")

    listed = list_saved(WebConfig(saved_dir=tmp_path))

    assert [(p["file"], p["name"], p["reviews"]) for p in listed] == [
        ("a.json", "a", 1), ("crawled/1_2.json", "1_2 (crawled)", 2)]


if __name__ == "__main__":
    import pathlib
    import tempfile

    test_crawled_products_are_listed_after_saved_ones(pathlib.Path(tempfile.mkdtemp()))
    print("ok")
