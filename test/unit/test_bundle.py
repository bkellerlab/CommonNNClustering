import numpy as np
import pytest

from commonnn.report import Summary
from commonnn import _bundle
from commonnn._primitive_types import P_AVALUE
from commonnn import _types


def test_bundle_init(file_regression):
    bundle = _bundle.Bundle()
    assert bundle.input_data is None
    assert bundle.graph is None
    assert bundle.labels is None
    assert bundle.root_indices is None
    assert bundle.parent_indices is None
    assert bundle.parent is None
    assert bundle.children == {}
    assert bundle.meta is None
    assert isinstance(bundle.summary, Summary)
    file_regression.check(f"{bundle!r}")


def test_bundle_init_parent():
    parent = _bundle.Bundle()
    other = _bundle.Bundle()
    child = _bundle.Bundle(alias="child", parent=parent)
    assert child.parent == parent
    assert child.parent != other
    assert child.hierarchy_level == 1
    assert parent.alias != child.alias


def test_bundle_input_data():
    input_data = []
    with pytest.raises(TypeError):
        _bundle.Bundle(input_data)

    input_data = _types.InputDataExtComponentsMemoryview(
        np.array([[0, 0, 0], [1, 2, 3]], order="c", dtype=P_AVALUE)
    )
    bundle = _bundle.Bundle(input_data)
    np.asarray(input_data.data)[0, 0] = 1
    np.testing.assert_array_equal(
        np.array(bundle.input_data), np.array(input_data.data)
    )


def test_bundle_labels():
    labels = _types.Labels.from_length(5)
    indices = _types.ReferenceIndices(np.arange(5), np.arange(5))
    bundle = _bundle.Bundle(labels=labels, reference_indices=indices)
    np.testing.assert_array_equal(bundle.labels, labels.labels)
    np.testing.assert_array_equal(bundle.root_indices, indices.root)
    np.testing.assert_array_equal(bundle.parent_indices, indices.parent)

    bundle.labels = np.array([1, 2, 3])
    assert isinstance(bundle._labels, _types.Labels)


def test_bundle_children():
    with pytest.raises(TypeError):
        _bundle.Bundle(children=[])

    bundle = _bundle.Bundle(
        children={
            1: _bundle.Bundle(
                alias="1", children={
                    1: _bundle.Bundle(alias="1.1")
                }
            ),
            2: _bundle.Bundle(alias="2")
        }
    )

    assert len(bundle.children) == 2
    assert bundle.get_child("1").alias == "1"
    assert bundle.get_child("1.1").alias == "1.1"
    assert bundle.get_child(2).alias == "2"

    with pytest.raises(KeyError):
        bundle.get_child(3)

    with pytest.raises(KeyError):
        bundle.get_child("1.3")
