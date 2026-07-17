from battery_soh.data.splits import assert_no_cell_leakage, make_cell_level_split


def test_cell_level_split_has_no_overlap():
    split = make_cell_level_split([f"cell_{i}" for i in range(10)], seed=7)
    assert_no_cell_leakage(split.to_dict())
    assert len(split.train) + len(split.val) + len(split.test) == 10


def test_cell_level_split_rejects_duplicates():
    try:
        make_cell_level_split(["a", "a"], seed=1)
    except ValueError as exc:
        assert "duplicates" in str(exc)
    else:
        raise AssertionError("Expected duplicate cell IDs to be rejected.")

