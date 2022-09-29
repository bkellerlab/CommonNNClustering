from cmath import exp
import numpy as np
import pytest

from commonnn._primitive_types import P_AINDEX, P_ABOOL
from commonnn._types import Labels, ReferenceIndices
from commonnn._bundle import Bundle


@pytest.mark.parametrize(
    "labels,consider,meta,from_sequence",
    [
        (np.zeros(10, dtype=P_AINDEX), None, None, False),
        (np.zeros(10, dtype=P_AINDEX), np.ones(10, dtype=P_ABOOL), None, False),
        (np.arange(10, dtype=P_AINDEX), None, {"origin": "made up"}, False),
        pytest.param(
            np.zeros(10, dtype=P_AINDEX), np.ones(9, dtype=P_ABOOL), None, False,
            marks=[pytest.mark.raises(exception=ValueError)]
        ),
        pytest.param(
            None, None, None, False,
            marks=[pytest.mark.raises(exception=TypeError)]
        ),
        pytest.param(
            1, None, None, False,
            marks=[pytest.mark.raises(exception=TypeError)]
        ),
        pytest.param(
            1, None, None, True,
            marks=[pytest.mark.raises(exception=ValueError)]
        ),
        ([2, 2, 1], None, None, True),
        ([2, 2, 1], [0, 0, 1], None, True),
    ]
)
def test_labels_init(
        labels, consider, meta, from_sequence, file_regression):
    if not from_sequence:
        _labels = Labels(labels, consider=consider, meta=meta)
    else:
        _labels = Labels.from_sequence(labels, consider=consider, meta=meta)

    assert isinstance(_labels.labels, np.ndarray)
    assert isinstance(_labels.consider, np.ndarray)
    assert isinstance(_labels.meta, dict)

    repr_ = f"{_labels!r}"
    str_ = f"{_labels!s}"
    file_regression.check(f"{repr_}\n{str_}\n{_labels.consider}\n{_labels.meta}")


def test_labels_from_length():
    _labels = Labels.from_length(5)
    assert _labels.n_points == 5


def test_labels_mapping():
    _labels = Labels.from_sequence([1, 1, 2, 2, 2, 0, 3])
    assert _labels.mapping == {0: [5], 1: [0, 1], 2: [2, 3, 4], 3: [6]}
    assert _labels.set == _labels.mapping.keys()


def test_labels_consider_set():
    _labels = Labels.from_sequence([1, 1, 1])
    assert _labels.consider_set == set()
    _labels.consider_set = [0, 1]
    assert _labels.consider_set == {0, 1}


@pytest.mark.parametrize(
    "member_cutoff,max_clusters,expected",
    [
        (None, None, np.array([2, 2, 1, 1, 1, 0, 0])),
        (1, None, np.array([2, 2, 1, 1, 1, 0, 3])),
        (None, 1, np.array([0, 0, 1, 1, 1, 0, 0])),

    ]
)
def test_labels_sort_by_size(member_cutoff, max_clusters, expected):
    original = [1, 1, 2, 2, 2, 0, 3]
    meta = {"params": {1: 1, 2: 2, 3: 3}}
    _labels = Labels.from_sequence(original, meta={k: v for k, v in meta.items()})
    _labels.sort_by_size(member_cutoff, max_clusters)
    assert np.all(_labels.labels == expected)
    for c, i in enumerate(_labels.labels):
        if i == 0:
            continue
        assert _labels.meta["params"][i] == meta["params"][original[c]]


def test_labels_sort_by_size_bundle():
    bundle = Bundle()
    bundle._children = {
        1: Bundle(alias="1"), 2: Bundle(alias="2"), 3: Bundle(alias="3")
    }
    _labels = Labels.from_sequence([1, 1, 2, 2, 2, 0, 3])
    _labels.sort_by_size(bundle=bundle)

    assert bundle._children[1].alias == "2"
    assert bundle._children[2].alias == "1"
    assert 3 not in bundle._children


def test_reference_indices_init():

    refindices = ReferenceIndices(np.arange(5), np.arange(5))

    assert refindices.root.shape[0] == 5
    assert refindices.parent.shape[0] == 5
