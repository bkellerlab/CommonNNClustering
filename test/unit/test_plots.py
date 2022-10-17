import json

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    MPL_FOUND = True
except ModuleNotFoundError:
    MPL_FOUND = False

try:
    import networkx as nx
    from networkx.algorithms.traversal.depth_first_search import dfs_postorder_nodes
    NX_FOUND = True
except ModuleNotFoundError:
    NX_FOUND = False

import pytest

from commonnn import plot


@pytest.mark.parametrize(
    "case_key",
    [
        pytest.param("empty", marks=[pytest.mark.raises(exception=AttributeError)]),
        "hierarchical_a"
    ]
)
def test_get_pieces(case_key, registered_clustering):
    pieces = plot.get_pieces(registered_clustering._bundle)
    assert len(set([sum(x[1] for x in y) for y in pieces])) == 1


@pytest.mark.networkx
def test_traverse_graph_dfs_children_first():
    if not NX_FOUND:
        pytest.skip("Test function requires networkx")

    g = nx.DiGraph({0: [1, 2], 1: [3, 4], 2: [5, 6]})
    node_list = list(plot.traverse_graph_dfs_children_first(g, 0))
    expected_node_list = list(dfs_postorder_nodes(g, source=0))

    assert node_list == expected_node_list


@pytest.mark.networkx
def test_find_node_positions_sugiyama_straight(file_regression):

    if not NX_FOUND:
        pytest.skip("Test function requires networkx")

    g = nx.DiGraph(
        {
            "1": ["1.1", "1.2"],
            "1.1": ["1.1.1", "1.1.2"],
            "1.1.2": ["1.1.2.1", "1.1.2.2"]
        }
    )
    positions = plot.find_node_positions_sugiyama_straight(g)
    positions = {k: list(v) for k, v in positions.items()}
    file_regression.check(json.dumps(positions, indent=4))


@pytest.mark.matplotlib
@pytest.mark.image_regression
@pytest.mark.parametrize(
    "case_key",
    ["hierarchical_a"]
)
def test_tree(
        case_key,
        registered_clustering,
        datadir,
        image_regression):

    if not MPL_FOUND:
        pytest.skip("Test function requires networkx")

    mpl.use("agg")

    fig, ax = plt.subplots()

    registered_clustering.tree(ax)
    fig.tight_layout()
    figname = datadir / f"tree_{case_key}.png"
    fig.savefig(figname)
    image_regression.check(
        figname.read_bytes(),
        basename=f"test_tree_{case_key}",
        diff_threshold=5,
    )


@pytest.mark.matplotlib
@pytest.mark.image_regression
@pytest.mark.parametrize(
    "case_key",
    ["hierarchical_a"]
)
def test_pie(
        case_key,
        registered_clustering,
        datadir,
        image_regression):

    if not MPL_FOUND:
        pytest.skip("Test function requires networkx")

    mpl.use("agg")

    fig, ax = plt.subplots()
    registered_clustering.pie(ax)
    fig.tight_layout()
    figname = datadir / f"pie_{case_key}.png"
    fig.savefig(figname)
    image_regression.check(
        figname.read_bytes(),
        basename=f"test_pie_{case_key}",
        diff_threshold=5,
    )
