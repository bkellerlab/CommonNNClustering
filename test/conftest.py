import pytest
import numpy as np

try:
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler
    SKLEARN_FOUND = True
except ModuleNotFoundError:
    SKLEARN_FOUND = False

from commonnn import cluster
from commonnn._primitive_types import P_AINDEX
from commonnn._bundle import Bundle
from commonnn._types import Labels, ReferenceIndices


@pytest.fixture
def toy_data_points(request):
    if not SKLEARN_FOUND:
        raise ModuleNotFoundError(
            "No module named 'sklearn'"
        )

    n_samples = request.node.funcargs.get("n_samples")
    gen_func = request.node.funcargs.get("gen_func")
    gen_kwargs = request.node.funcargs.get("gen_kwargs")

    generation_functions = {
        "moons": datasets.make_moons,
        "blobs": datasets.make_blobs
    }

    points, reference_labels = generation_functions[gen_func](
        n_samples=n_samples, **gen_kwargs
    )

    points = StandardScaler().fit_transform(points)
    reference_labels += 1
    return points, reference_labels


def make_empty_clustering():
    return cluster.Clustering()


def make_hierarchical_clustering_a():
    bundle = Bundle(
        labels=Labels(
            np.array([0, 0, 1, 1, 0, 0, 1, 2, 1, 1, 1, 2, 2, 1, 0], dtype=P_AINDEX)
        )
    )
    clustering = cluster.Clustering(bundle)

    for i in [0, 1, 2]:
        bundle.add_child(i)

    bundle._children[1]._labels = Labels(
        np.array([0, 1, 0, 2, 2, 2, 1], dtype=P_AINDEX)
    )
    bundle._children[1]._reference_indices = ReferenceIndices(
        np.array([2, 3, 6, 8, 9, 10, 13]),
        np.array([2, 3, 6, 8, 9, 10, 13])
    )

    for i in [0, 1, 2]:
        bundle.get_child(1).add_child(i)

    bundle.get_child([1, 2])._labels = Labels(
        np.array([2, 1, 0], dtype=P_AINDEX)
    )
    bundle.get_child([1, 2])._reference_indices = ReferenceIndices(
        np.array([8, 9, 10]),
        np.array([3, 4, 5])
    )

    return clustering


def make_trivial_clustering():

    bundle = Bundle(
        labels=Labels(
            np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=P_AINDEX)
        )
    )
    clustering = cluster.Clustering(bundle)

    return clustering


registered_clustering_map = {
    "empty": make_empty_clustering,
    "hierarchical_a": make_hierarchical_clustering_a,
    "trivial": make_trivial_clustering,
}


@pytest.fixture
def registered_clustering(request):
    key = request.node.funcargs.get("case_key")
    return registered_clustering_map[key]()
