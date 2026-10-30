from src.data_handling.datasets import get_data_paths, split_data


def _items(region, n):
    return [{"image": f"data/Task1/{region}/1{region}A{i:03d}/ct.mha", "mask": ""} for i in range(n)]


def test_split_is_stratified_disjoint_and_deterministic():
    data = _items("AB", 50) + _items("HN", 30) + _items("TH", 20)

    train, val, test = split_data(data, 0.7, 0.1, 0.2, random_seed=42)

    images = lambda split: {item["image"] for item in split}
    assert images(train).isdisjoint(images(val))
    assert images(train).isdisjoint(images(test))
    assert images(val).isdisjoint(images(test))
    assert len(train) + len(val) + len(test) == len(data)
    # 70 % of every region goes to training
    assert sum("/AB/" in item["image"] for item in train) == 35
    assert sum("/HN/" in item["image"] for item in train) == 21
    assert sum("/TH/" in item["image"] for item in train) == 14

    assert split_data(data, 0.7, 0.1, 0.2, random_seed=42) == (train, val, test)


def test_get_data_paths_skips_overview_and_missing_dirs(tmp_path):
    region = tmp_path / "Task1" / "AB"
    for name in ("1ABA001", "1ABA002", "overview"):
        (region / name).mkdir(parents=True)

    _, cts, masks = get_data_paths([region, tmp_path / "missing"])

    assert cts == [str(region / "1ABA001" / "ct.mha"), str(region / "1ABA002" / "ct.mha")]
    assert masks == [str(region / "1ABA001" / "mask.mha"), str(region / "1ABA002" / "mask.mha")]
