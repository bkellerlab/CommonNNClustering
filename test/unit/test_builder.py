import numpy as np
import pytest

from commonnn import recipes
from commonnn import _fit, _types
from commonnn._primitive_types import P_AVALUE


def test_init_builder():
    builder = recipes.Builder({})
    assert builder.recipe == {}

    builder = recipes.Builder()
    expected_recipe = {
        ".".join(
            recipes.COMPONENT_ALT_KW_MAP.get(kw, kw)
            for kw in k.split(".")
        ): v
        for k, v in recipes.get_registered_recipe(builder.default_recipe).items()
    }

    assert expected_recipe == builder.recipe

    builder = recipes.Builder(
        prep="array",
        fitter__ngetter__dgetter="lookup"
    )

    expected_recipe["preparation_hook"] = "array"
    expected_recipe["fitter.neighbours_getter.distance_getter"] = "lookup"

    assert expected_recipe == builder.recipe


@pytest.mark.parametrize(
    "recipe,make,expected",
    [
        ({}, "queue", object),
        ({"queue": None}, "queue", None),
        ({"queue": "fifo"}, "queue", _types.QueueExtFIFOQueue),
        ({"queue": _types.QueueFIFODeque}, "queue", _types.QueueFIFODeque),
        ({"queue": (_types.QueueExtLIFOVector, ([1, 2, 3],), {})}, "queue", _types.QueueExtLIFOVector),
        ({"pq": _types.PriorityQueueMaxHeap}, "priority_queue", _types.PriorityQueueMaxHeap),
        ({"checker": "contains"}, "similarity_checker", _types.SimilarityCheckerExtContains),
        ({"checker": "switch"}, "similarity_checker", _types.SimilarityCheckerExtSwitchContains),
        ({"checker": "screen"}, "similarity_checker", _types.SimilarityCheckerExtScreensorted),
        ({"checker": _types.SimilarityCheckerContains}, "similarity_checker", _types.SimilarityCheckerContains),
        ({"checker": _types.SimilarityCheckerSwitchContains}, "similarity_checker", _types.SimilarityCheckerSwitchContains),
        ({"metric": "dummy"}, "metric", _types.MetricExtDummy),
        ({"metric": _types.MetricDummy}, "metric", _types.MetricDummy),
        ({"metric": "precomputed"}, "metric", _types.MetricExtPrecomputed),
        ({"metric": "euclidean"}, "metric", _types.MetricExtEuclidean),
        ({"metric": "euclidean_r"}, "metric", _types.MetricExtEuclideanReduced),
        ({"metric": ("euclidean_periodic_r", (), {"bounds": np.array([1, 2], dtype=P_AVALUE)})}, "metric", _types.MetricExtEuclideanPeriodicReduced),
        ({"na": "vector"}, "neighbours", _types.NeighboursExtVector),
        ({"na": _types.NeighboursList}, "neighbours", _types.NeighboursList),
        ({"na": _types.NeighboursSet}, "neighbours", _types.NeighboursSet),
        ({"nb": "uset"}, "neighbour_neighbours", _types.NeighboursExtUnorderedSet),
        ({"nb": "vuset"}, "neighbour_neighbours", _types.NeighboursExtVectorUnorderedSet),
        (
            {"dgetter": "metric", "dgetter.metric": "dummy"},
            "distance_getter", _types.DistanceGetterExtMetric
        ),
        (
            {"dgetter": _types.DistanceGetterMetric, "dgetter.metric": "dummy"},
            "distance_getter", _types.DistanceGetterMetric
        ),
        ({"dgetter": "lookup"}, "distance_getter", _types.DistanceGetterExtLookup),
        ({"dgetter": _types.DistanceGetterLookup}, "distance_getter", _types.DistanceGetterLookup),
        (
            {"ngetter": "brute_force", "ngetter.dgetter": "metric", "ngetter.dgetter.metric": "dummy"},
            "neighbours_getter", _types.NeighboursGetterExtBruteForce
        ),
        (
            {"ngetter": _types.NeighboursGetterBruteForce, "ngetter.dgetter": "metric", "ngetter.dgetter.metric": "dummy"},
            "neighbours_getter", _types.NeighboursGetterBruteForce
        ),
        ({"ngetter": "lookup"}, "neighbours_getter", _types.NeighboursGetterExtLookup),
        ({"ngetter": _types.NeighboursGetterLookup}, "neighbours_getter", _types.NeighboursGetterLookup),
        ({"ngetter": _types.NeighboursGetterRecomputeLookup}, "neighbours_getter", _types.NeighboursGetterRecomputeLookup),
        (
            {
                "fitter": "bfs", "fitter.na": "vector", "fitter.checker": "contains", "fitter.q": "fifo",
                "fitter.ngetter": "brute_force", "fitter.ngetter.dgetter": "metric", "fitter.ngetter.dgetter.metric": "dummy"
            },
            "fitter", _fit.FitterExtCommonNNBFS
        ),
        (
            {
                "fitter": _fit.FitterCommonNNBFS, "fitter.na": "vector", "fitter.checker": "contains", "fitter.q": "fifo",
                "fitter.ngetter": "brute_force", "fitter.ngetter.dgetter": "metric", "fitter.ngetter.dgetter.metric": "dummy"
            },
            "fitter", _fit.FitterCommonNNBFS
        ),
        (
            {
                "fitter": _fit.FitterCommonNNBFSDebug, "fitter.na": "vector", "fitter.checker": "contains", "fitter.q": "fifo",
                "fitter.ngetter": "brute_force", "fitter.ngetter.dgetter": "metric", "fitter.ngetter.dgetter.metric": "dummy"
            },
            "fitter", _fit.FitterCommonNNBFSDebug
        ),
        (
            {
                "hierarchical_fitter": "repeat",
                "hierarchical_fitter.fitter": "bfs", "hierarchical_fitter.fitter.na": "vector",
                "hierarchical_fitter.fitter.checker": "contains", "hierarchical_fitter.fitter.q": "fifo",
                "hierarchical_fitter.fitter.ngetter": "brute_force", "hierarchical_fitter.fitter.ngetter.dgetter": "metric",
                "hierarchical_fitter.fitter.ngetter.dgetter.metric": "dummy"
            },
            "hierarchical_fitter", _fit.HierarchicalFitterRepeat
        ),
        (
            {
                "hierarchical_fitter": "mst_debug",
                "hierarchical_fitter.na": "vector",
                "hierarchical_fitter.checker": "contains",
                "hierarchical_fitter.prioq": "maxheap",
                "hierarchical_fitter.ngetter": "lookup"
            },
            "hierarchical_fitter", _fit.HierarchicalFitterCommonNNMSTPrim
        ),
        (
            {
                "hierarchical_fitter": "mst",
                "hierarchical_fitter.na": "vector",
                "hierarchical_fitter.checker": "contains",
                "hierarchical_fitter.prioq": "maxheap",
                "hierarchical_fitter.ngetter": "lookup"
            },
            "hierarchical_fitter", _fit.HierarchicalFitterExtCommonNNMSTPrim
        ),
        (
            {
                "predictor": _fit.PredictorCommonNNFirstmatch,
                "predictor.na": "vector",
                "predictor.checker": "contains",
                "predictor.ngetter": "lookup"
            },
            "predictor", _fit.PredictorCommonNNFirstmatch
        ),
    ]
)
def test_make_components(recipe, make, expected):
    builder = recipes.Builder(recipe)
    component = builder.make_component(make)
    assert (component is None) or (isinstance(component, expected))


@pytest.mark.parametrize(
    "recipe,data_kind,preparation_hook,expected",
    [
        pytest.param(
            {}, "components", None, None,
            marks=[pytest.mark.raises(exception=LookupError)]
        ),
        (
            {"input": _types.InputDataComponentsSequence}, "components", None,
            _types.InputDataComponentsSequence
        ),
        (
            {"input": (_types.InputDataSklearnKDTree, (), {"leaf_size": 10})}, "components", "components_array_from_parts",
            _types.InputDataSklearnKDTree
        ),
        (
            {"input": "components_mview"}, "components", "components_array_from_parts",
            _types.InputDataExtComponentsMemoryview
        ),
        (
            {"input": ("distances_mview", (), {}), "prep": "components_array_from_parts"}, "distances", None,
            _types.InputDataExtDistancesMemoryview
        ),
        (
            {"input": "neighbourhoods_mview"}, "neighbourhoods", "padded_neighbourhoods_array",
            _types.InputDataExtNeighbourhoodsMemoryview
        ),
    ]
)
def test_make_input_data(
        recipe, data_kind, preparation_hook, expected,
        basic_components, basic_distances, basic_neighbourhoods):
    if data_kind == "components":
        data = basic_components
    elif data_kind == "distances":
        data = basic_distances
    elif data_kind == "neighbourhoods":
        data = basic_neighbourhoods
    else:
        raise ValueError

    builder = recipes.Builder(recipe)
    input_data = builder.make_input_data(data, preparation_hook=preparation_hook)
    assert isinstance(input_data, expected)
