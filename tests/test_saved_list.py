import json

from vi_web.analyze import list_saved
from vi_web.config import WebConfig


def test_crawled_products_are_listed_after_saved_ones(tmp_path):
    (tmp_path / "crawled").mkdir()
    (tmp_path / "a.json").write_text(json.dumps([{"comment": "tốt"}]), encoding="utf-8")
    two = json.dumps([{"comment": "ok"}, {"comment": "kém"}])
    (tmp_path / "crawled" / "Ban-phim-F75_1_2.json").write_text(two, encoding="utf-8")
    (tmp_path / "crawled" / "3_4.json").write_text(two, encoding="utf-8")  # a link with no name: the ids stay

    listed = list_saved(WebConfig(saved_dir=tmp_path))

    assert [(p["file"], p["name"], p["reviews"]) for p in listed] == [
        ("a.json", "a", 1), ("crawled/3_4.json", "3_4 (crawled)", 2),
        ("crawled/Ban-phim-F75_1_2.json", "Ban phim F75 (crawled)", 2)]


if __name__ == "__main__":
    import pathlib
    import tempfile

    test_crawled_products_are_listed_after_saved_ones(pathlib.Path(tempfile.mkdtemp()))
    print("ok")
