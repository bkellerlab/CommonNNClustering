try:
    import networkx as nx
    NX_FOUND = True
except ModuleNotFoundError:
    NX_FOUND = False

import numpy as np
import pytest

from commonnn.report import Summary
from commonnn import _bundle
from commonnn._primitive_types import P_AVALUE, P_AINDEX
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
    assert bundle.meta == {}
    assert isinstance(bundle.summary, Summary)
    file_regression.check(f"{bundle!r}\n\n{bundle.info()}")


def test_bundle_init_parent():
    parent = _bundle.Bundle()
    other = _bundle.Bundle()
    child = _bundle.Bundle(alias="child", parent=parent)
    assert child.parent == parent
    assert child.parent != other
    assert child.hierarchy_level == 1
    assert parent.alias != child.alias


def test_bundle_input_data(file_regression):
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

    file_regression.check(f"{bundle!r}\n\n{bundle.info()}")


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

    assert bundle.get_child([1, 1]) == bundle[1, 1]

    bundle.add_child(3)
    assert bundle[3].alias == "root"

    bundle["4"] = _bundle.Bundle(alias=4)
    assert bundle[4].alias == "4"

    bundle.add_child("3.1", bundle=_bundle.Bundle(alias="3.1"))
    assert bundle["3.1"].alias == "3.1"


def test_bundle_summary():
    bundle = _bundle.Bundle()

    with pytest.raises(TypeError):
        bundle.summary = set()

    bundle.summary = []
    bundle.summary.append(1)
    assert bundle.summary[0] == 1


@pytest.mark.parametrize(
    "input_data_type,data,meta,labels,root_indices,parent_indices",
    [
        (
            _types.InputDataExtComponentsMemoryview,
            np.array(
                [[0, 0, 0],
                 [1, 1, 1]],
                order="C", dtype=P_AVALUE
            ),
            None,
            np.array([1, 2], dtype=P_AINDEX),
            None, None
        ),
        (
            _types.InputDataExtComponentsMemoryview,
            np.array(
                [[0, 0], [1, 1], [2, 2],
                 [3, 3], [4, 4], [5, 5],
                 [6, 6], [7, 7], [8, 8]],
                order="c", dtype=P_AVALUE
            ),
            {"edges": []},
            np.array([2, 1, 2, 2, 2, 1, 0, 1, 0], dtype=P_AINDEX),
            None, None
        ),
        (
            _types.InputDataExtComponentsMemoryview,
            np.array(
                [[0, 0, 0],
                 [1, 1, 1],
                 [2, 2, 2],
                 [3, 3, 3],
                 [4, 4, 4],
                 [5, 5, 5]],
                order="C", dtype=P_AVALUE
            ),
            {"edges": [2, 2, 2]},
            np.array([1, 2, 1, 2, 0, 0], dtype=P_AINDEX),
            None, None
        ),
        (
            _types.InputDataExtComponentsMemoryview,
            np.array(
                [[1, 1, 1],
                 [2, 2, 2],
                 [3, 3, 3],
                 [4, 4, 4]],
                order="C", dtype=P_AVALUE
            ),
            {"edges": [4]},
            np.array([1, 2, 1, 2], dtype=P_AINDEX),
            np.array([1, 2, 3, 4], dtype=P_AINDEX),
            np.array([1, 2, 3, 4], dtype=P_AINDEX),
        ),
        (
            _types.InputDataExtComponentsMemoryview,
            np.array(
                [[1, 1, 1],
                 [4, 4, 4]],
                order="C", dtype=P_AVALUE
            ),
            {"edges": [1, 1]},
            np.array([1, 2], dtype=P_AINDEX),
            np.array([1, 3], dtype=P_AINDEX),
            np.array([0, 2], dtype=P_AINDEX),
        ),
    ]
)
def test_isolate(
        input_data_type, data, meta, labels, root_indices, parent_indices,
        file_regression):

    if root_indices is not None:
        reference_indices = _types.ReferenceIndices(root_indices, parent_indices)
    else:
        reference_indices = None

    bundle = _bundle.Bundle(
        input_data_type(data, meta=meta),
        labels=labels,
        reference_indices=reference_indices
    )

    bundle.isolate(isolate_input_data=False)
    label_set = bundle._labels.set
    label_set.discard(0)
    assert len(bundle.children) == len(label_set)
    assert bundle[1]._input_data is None

    bundle.isolate()
    label_mapping = bundle._labels.mapping
    for label, indices in label_mapping.items():
        if label == 0: continue
        assert len(indices) == bundle[label]._input_data.n_points

    report = ""
    for label in label_set:
        isolated_points = bundle._children[label]._input_data

        edges = isolated_points.meta.get('edges', "None")
        report += (
            f"Child {label}\n"
            f'{"=" * 80}\n'
            f"Data:\n{isolated_points.to_components_array()}\n\n"
            f"Edges:\n{edges}\n\n"
            f"Root:\n{bundle._children[label].root_indices}\n\n"
            f"Parent:\n{bundle._children[label].parent_indices}\n\n"
        )

    report = report.rstrip("\n")

    file_regression.check(report)


@pytest.mark.parametrize(
    "case_key,depth,expected,expected_params",
    [
        (
            "hierarchical_a", 1,
            [0, 0, 0, 3, 0, 0, 0, 2, 4, 4, 4, 2, 2, 3, 0],
            {0: (1., 1), 2: (1., 1), 3: (1., 2), 4: (1., 2)}
        ),
        (
            "hierarchical_a", None,
            [0, 0, 0, 3, 0, 0, 0, 2, 6, 5, 0, 2, 2, 3, 0],
            {0: (1., 1), 2: (1., 1), 3: (1., 2), 5: (1., 3), 6: (1., 3)}
        )
    ]
)
def test_reel(case_key, registered_clustering, depth, expected, expected_params):
    registered_clustering.reel(depth=depth)
    np.testing.assert_array_equal(
        registered_clustering._bundle.labels,
        expected
    )
    assert registered_clustering._bundle._labels.meta["params"] == expected_params


def test_bfs():
    root = _bundle.Bundle(
        children={
            1: _bundle.Bundle(alias="1"),
            2: _bundle.Bundle(alias="2"),
            3: _bundle.Bundle(
                alias="3",
                children={
                    1: _bundle.Bundle(alias="3.1"),
                    2: _bundle.Bundle(alias="3.2"),
                }
            ),
        }
    )

    expected = ["root", "1", "2", "3", "3.1", "3.2"]
    got = list(b.alias for b in _bundle.bfs(root))
    assert got == expected

    expected = ["1", "2", "3.1", "3.2"]
    got = list(b.alias for b in _bundle.bfs_leafs(root))
    assert got == expected


def test_check_children():
    if not NX_FOUND:
        pytest.skip("Test function requires pandas")

    root = _bundle.Bundle(
        alias="root",
        children={
            1: _bundle.Bundle(
                alias="pseudo_root",
                graph=nx.Graph(),
                children={
                    1: _bundle.Bundle(alias="1", graph=nx.Graph()),
                    2: _bundle.Bundle(alias="2", graph=nx.Graph()),
                    3: _bundle.Bundle(
                        alias="3",
                        graph=nx.Graph(),
                        children={
                            1: _bundle.Bundle(alias="3.1", graph=nx.Graph()),
                            2: _bundle.Bundle(
                                alias="3.2",
                                graph=nx.Graph(),
                                children={
                                    1: _bundle.Bundle(alias="3.2.1", graph=nx.Graph()),
                                    2: _bundle.Bundle(alias="3.2.2", graph=nx.Graph()),
                                }
                            ),
                        }
                    ),
                }
            )
        }
    )

    root["1.3.2.1"]._lambda = 5
    root["1.3.2.2"]._lambda = 5
    root["1.3.1"]._lambda = 4
    root["1.3.2"]._lambda = 2
    root["1.3"]._lambda = 2
    root["1.2"]._lambda = 3
    root["1.1"]._lambda = 3
    root["1"]._lambda = 2
    root._lambda = 2

    _bundle.check_children(root, member_cutoff=0, needs_folding=True)

    expected = ["root", "1", "2", "3.1", "3.2.1", "3.2.2"]
    got = list(b.alias for b in _bundle.bfs(root))
    assert got == expected

    _bundle.check_children(root, member_cutoff=1, needs_folding=True)

    expected = ["root"]
    got = list(b.alias for b in _bundle.bfs(root))
    assert got == expected


def test_check_children_remove_lonechild():
    if not NX_FOUND:
        pytest.skip("Test function requires pandas")

    root = _bundle.Bundle(
        alias="root",
        children={
            1: _bundle.Bundle(
                alias="1",
                graph=nx.Graph(),
                children={
                    1: _bundle.Bundle(alias="1.1", graph=nx.Graph()),
                    2: _bundle.Bundle(alias="1.2", graph=nx.Graph()),
                }
            )
        }
    )

    _bundle.check_children(root, member_cutoff=0, needs_folding=False)
    expected = ["1.1", "1.2"]
    got = [b.alias for b in root.children.values()]
    assert got == expected
