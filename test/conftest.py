import pytest
import numpy as np

try:
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler
    SKLEARN_FOUND = True
except ModuleNotFoundError:
    SKLEARN_FOUND = False

# from commonnn import cluster
from commonnn._primitive_types import P_AINDEX
# from commonnn._bundle import Bundle
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


@pytest.fixture
def basic_components():
    return [   # point index
        [0, 0],       # 0
        [1, 1],       # 1
        [1, 0],       # 2
        [0, -1],      # 3
        [0.5, -0.5],  # 4
        [2, 1.5],    # 5
        [2.5, -0.5],  # 6
        [4, 2],       # 7
        [4.5, 2.5],   # 8
        [5, -1],      # 9
        [5.5, -0.5],  # 10
        [5.5, -1.5],  # 11
    ]

@pytest.fixture
def basic_distances():
    return [
        [0.  , 1.41, 1.  , 1.  , 0.71, 2.5 , 2.55, 4.47, 5.15, 5.1 , 5.52, 5.7 ],
        [1.41, 0.  , 1.  , 2.24, 1.58, 1.12, 2.12, 3.16, 3.81, 4.47, 4.74, 5.15],
        [1.  , 1.  , 0.  , 1.41, 0.71, 1.8 , 1.58, 3.61, 4.3 , 4.12, 4.53, 4.74],
        [1.  , 2.24, 1.41, 0.  , 0.71, 3.2 , 2.55, 5.  , 5.7 , 5.  , 5.52, 5.52],
        [0.71, 1.58, 0.71, 0.71, 0.  , 2.5 , 2.  , 4.3 , 5.  , 4.53, 5.  , 5.1 ],
        [2.5 , 1.12, 1.8 , 3.2 , 2.5 , 0.  , 2.06, 2.06, 2.69, 3.91, 4.03, 4.61],
        [2.55, 2.12, 1.58, 2.55, 2.  , 2.06, 0.  , 2.92, 3.61, 2.55, 3.  , 3.16],
        [4.47, 3.16, 3.61, 5.  , 4.3 , 2.06, 2.92, 0.  , 0.71, 3.16, 2.92, 3.81],
        [5.15, 3.81, 4.3 , 5.7 , 5.  , 2.69, 3.61, 0.71, 0.  , 3.54, 3.16, 4.12],
        [5.1 , 4.47, 4.12, 5.  , 4.53, 3.91, 2.55, 3.16, 3.54, 0.  , 0.71, 0.71],
        [5.52, 4.74, 4.53, 5.52, 5.  , 4.03, 3.  , 2.92, 3.16, 0.71, 0.  , 1.  ],
        [5.7 , 5.15, 4.74, 5.52, 5.1 , 4.61, 3.16, 3.81, 4.12, 0.71, 1.  , 0.  ]
    ]


@pytest.fixture
def basic_neighbourhoods():
    return [
        [0, 2, 3, 4],
        [1, 2],
        [0, 1, 2, 4],
        [0, 3, 4],
        [0, 2, 3, 4],
        [5],
        [6],
        [7, 8],
        [7, 8],
        [9, 10, 11],
        [9, 10, 11],
        [9, 10, 11]
    ]




# def make_empty_clustering():
#     return cluster.Clustering()
# 
# 
# def make_hierarchical_clustering_a():
#     bundle = Bundle(
#         labels=Labels(
#             np.array([0, 0, 1, 1, 0, 0, 1, 2, 1, 1, 1, 2, 2, 1, 0], dtype=P_AINDEX)
#         )
#     )
#     clustering = cluster.Clustering(bundle)
# 
#     for i in [0, 1, 2]:
#         bundle.add_child(i)
# 
#     bundle._children[1]._labels = Labels(
#         np.array([0, 1, 0, 2, 2, 2, 1], dtype=P_AINDEX)
#     )
#     bundle._children[1]._reference_indices = ReferenceIndices(
#         np.array([2, 3, 6, 8, 9, 10, 13]),
#         np.array([2, 3, 6, 8, 9, 10, 13])
#     )
# 
#     for i in [0, 1, 2]:
#         bundle.get_child(1).add_child(i)
# 
#     bundle.get_child([1, 2])._labels = Labels(
#         np.array([2, 1, 0], dtype=P_AINDEX)
#     )
#     bundle.get_child([1, 2])._reference_indices = ReferenceIndices(
#         np.array([8, 9, 10]),
#         np.array([3, 4, 5])
#     )
# 
#     return clustering
# 
# 
# def make_trivial_clustering():
# 
#     bundle = Bundle(
#         labels=Labels(
#             np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=P_AINDEX)
#         )
#     )
#     clustering = cluster.Clustering(bundle)
# 
#     return clustering
# 
# 
# registered_clustering_map = {
#     "empty": make_empty_clustering,
#     "hierarchical_a": make_hierarchical_clustering_a,
#     "trivial": make_trivial_clustering,
# }
# 
# 
# @pytest.fixture
# def registered_clustering(request):
#     key = request.node.funcargs.get("case_key")
#     return registered_clustering_map[key]()
