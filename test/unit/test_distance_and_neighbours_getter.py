import numpy as np
import pytest

from commonnn import recipes
from commonnn._primitive_types import P_AVALUE
from commonnn import _types


@pytest.mark.parametrize(
    "getter_type,args,kwargs",
    [
        (_types.DistanceGetterLookup, (), {}),
        (_types.DistanceGetterMetric, (_types.MetricDummy(),), {}),
        (_types.DistanceGetterExtLookup, (), {}),
        (_types.DistanceGetterExtMetric, (_types.MetricExtDummy(),), {}),
        (_types.NeighboursGetterBruteForce, (_types.DistanceGetterLookup(),), {}),
        (_types.NeighboursGetterExtBruteForce, (_types.DistanceGetterExtLookup(),), {}),
        (_types.NeighboursGetterLookup, (), {}),
        (_types.NeighboursGetterExtLookup, (), {}),
        (_types.NeighboursGetterRecomputeLookup, (), {})
    ]
)
def test_getter_init(getter_type, args, kwargs, file_regression):
    getter = getter_type(*args, **kwargs)
    file_regression.check(f"{getter!r}\n{getter!s}")

    if issubclass(getter_type, _types.NeighboursGetter):
        assert getter.is_selfcounting in (True, False)
        assert getter.is_sorted in (True, False)


@pytest.mark.parametrize(
    "dgetter_type",
    [_types.DistanceGetterLookup, _types.DistanceGetterExtLookup]
)
@pytest.mark.parametrize(
    "input_data_type,hook,hook_kwargs",
    [
        (
            _types.InputDataExtDistancesMemoryview,
            recipes.prepare_to_array, {"order": "c", "dtype": P_AVALUE},
        ),
    ]
)
def test_get_distances_lookup(
        input_data_type, hook, hook_kwargs, dgetter_type, basic_distances):

    if hook is None:
        hook = recipes.prepare_pass

    if hook_kwargs is None:
        hook_kwargs = {}

    data_args, data_kwargs = hook(basic_distances, **hook_kwargs)
    input_data = input_data_type(*data_args, **data_kwargs)
    distance_getter = dgetter_type()

    n_points = len(basic_distances)
    for i in range(n_points):
        for j in range(n_points):
            dsingle = distance_getter.get_single(i, j, input_data)
            dsingle_other = distance_getter.get_single_other(
                i, j, input_data, input_data
            )
            assert dsingle == dsingle_other == basic_distances[i][j]


@pytest.mark.parametrize(
    "dgetter_type,metric_type",
    [
        (
            _types.DistanceGetterMetric,
            _types.MetricEuclidean,
        ),
        (
            _types.DistanceGetterExtMetric,
            _types.MetricExtEuclidean,
        ),
    ]
)
@pytest.mark.parametrize(
    "input_data_type,hook,hook_kwargs",
    [
        (
            _types.InputDataExtComponentsMemoryview,
            recipes.prepare_to_array, {"order": "c", "dtype": P_AVALUE},
        ),
    ]
)
def test_get_distances_metric(
        input_data_type, hook, hook_kwargs, dgetter_type, metric_type,
        basic_components, basic_distances):

    if hook is None:
        hook = recipes.prepare_pass

    if hook_kwargs is None:
        hook_kwargs = {}

    data_args, data_kwargs = hook(basic_components, **hook_kwargs)
    input_data = input_data_type(*data_args, **data_kwargs)
    distance_getter = dgetter_type(metric=metric_type())

    n_points = len(basic_components)
    for i in range(n_points):
        for j in range(n_points):
            dsingle = distance_getter.get_single(i, j, input_data)
            dsingle_other = distance_getter.get_single_other(
                i, j, input_data, input_data
            )
            assert dsingle == dsingle_other
            np.testing.assert_approx_equal(
                dsingle, basic_distances[i][j], significant=2
            )


@pytest.mark.parametrize(
    "input_data_type,hook,neighbours_type,ngetter_type",
    [
        (
            _types.InputDataNeighbourhoodsSequence,
            None, _types.NeighboursList,
            _types.NeighboursGetterLookup
        ),
        (
            _types.InputDataExtNeighbourhoodsMemoryview,
            recipes.prepare_padded_neighbourhoods_array,
            _types.NeighboursExtVector,
            _types.NeighboursGetterExtLookup
        ),
    ]
)
def test_get_neighbours_lookup(
        input_data_type, hook, neighbours_type, ngetter_type, basic_neighbourhoods):

    if hook is None:
        hook = recipes.prepare_pass

    data_args, data_kwargs = hook(basic_neighbourhoods)
    input_data = input_data_type(*data_args, **data_kwargs)
    neighbours = neighbours_type()
    neighbours_getter = ngetter_type()
    cluster_params = _types.CommonNNParameters.from_mapping({
        "radius_cutoff": 0.0, "similarity_cutoff": 0,
    })

    n_points = len(basic_neighbourhoods)
    for i in range(n_points):
        neighbours_getter.get(i, input_data, neighbours, cluster_params)
        assert neighbours._n_points == len(basic_neighbourhoods[i])

        for n in range(neighbours._n_points):
            assert neighbours.get_member(n) == basic_neighbourhoods[i][n]

        neighbours_getter.get_other(
            i, input_data, input_data, neighbours, cluster_params
        )
        assert neighbours._n_points == len(basic_neighbourhoods[i])

        for n in range(neighbours._n_points):
            assert neighbours.get_member(n) == basic_neighbourhoods[i][n]


@pytest.mark.parametrize(
    "ngetter_type,dgetter_type,metric_type,input_data_type,hook,hook_kwargs",
    [
        (
            _types.NeighboursGetterRecomputeLookup,
            None, None,
            _types.InputDataSklearnKDTree,
            recipes.prepare_to_array, {"order": "c", "dtype": P_AVALUE},
        ),
        (
            _types.NeighboursGetterBruteForce,
            _types.DistanceGetterExtMetric,
            _types.MetricExtEuclidean,
            _types.InputDataExtComponentsMemoryview,
            recipes.prepare_to_array, {"order": "c", "dtype": P_AVALUE},
        ),
        (
            _types.NeighboursGetterExtBruteForce,
            _types.DistanceGetterExtMetric,
            _types.MetricExtEuclidean,
            _types.InputDataExtComponentsMemoryview,
            recipes.prepare_to_array, {"order": "c", "dtype": P_AVALUE},
        ),
    ]
)
def test_get_neighbours_dgetter(
        input_data_type, hook, hook_kwargs, ngetter_type, dgetter_type, metric_type,
        basic_components, basic_neighbourhoods):

    if hook is None:
        hook = recipes.prepare_pass

    if hook_kwargs is None:
        hook_kwargs = {}

    data_args, data_kwargs = hook(basic_components, **hook_kwargs)
    input_data = input_data_type(*data_args, **data_kwargs)

    dgetter_args = []
    if metric_type is not None:
        dgetter_args.append(metric_type())

    ngetter_args = []
    if dgetter_type is not None:
        ngetter_args.append(dgetter_type(*dgetter_args))
    ngetter = ngetter_type(*ngetter_args)

    neighbours = _types.NeighboursExtVector(initial_size=10)

    params = _types.CommonNNParameters.from_mapping(
        {"radius_cutoff": 1.1, "similarity_cutoff": 0}
    )

    n_points = len(basic_components)
    for i in range(n_points):
        expected = set(basic_neighbourhoods[i])
        ngetter.get(i, input_data, neighbours, params)
        nset = set(neighbours.get_member(x) for x in range(neighbours.n_points))
        assert nset == expected

    if hasattr(input_data, "_radius"):
        input_data._radius = 0

    for i in range(n_points):
        expected = set(basic_neighbourhoods[i])
        ngetter.get_other(i, input_data, input_data, neighbours, params)
        nset = set(neighbours.get_member(x) for x in range(neighbours.n_points))
        assert nset == expected
