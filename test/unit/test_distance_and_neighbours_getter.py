import pytest

from commonnn import recipes
from commonnn._primitive_types import P_AVALUE
from commonnn import _types


@pytest.mark.parametrize(
    "input_data_type,hook,hook_kwargs,dgetter_type",
    [
        (
            _types.InputDataExtDistancesMemoryview,
            recipes.prepare_to_array, {"order": "c", "dtype": P_AVALUE},
            _types.DistanceGetterExtLookup
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
            dsingle_other = distance_getter.get_single_other(i, j, input_data, input_data)
            assert dsingle == dsingle_other == basic_distances[i][j]


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
