import pytest
import numpy as np

from commonnn import recipes
from commonnn._primitive_types import P_AVALUE
from commonnn import _types


@pytest.mark.parametrize(
    "input_data_type,hook,kwargs",
    [
        (_types.InputDataComponentsSequence, None, None),
        (
            _types.InputDataExtComponentsMemoryview, recipes.prepare_to_array,
            {"order": "c", "dtype": P_AVALUE}
        ),
        (
            _types.InputDataSklearnKDTree, recipes.prepare_to_array,
            {"order": "c", "dtype": P_AVALUE}
        )
    ],
)
def test_input_data_init_components(
        input_data_type,
        hook, kwargs,
        file_regression,
        basic_components):

    n_points = len(basic_components)
    n_dim = len(basic_components[0])

    if hook is None:
        hook = recipes.prepare_pass

    if kwargs is None:
        kwargs = {}

    data_args, data_kwargs = hook(basic_components, **kwargs)
    meta = data_kwargs.get("meta", {})
    meta["dummy"] = "dummy"
    data_kwargs["meta"] = meta
    input_data = input_data_type(*data_args, **data_kwargs)

    assert input_data.meta["dummy"] == "dummy"
    assert input_data.data is not None
    assert input_data.n_points == n_points
    assert input_data.n_dim == n_dim

    for point in range(n_points):
        for dim in range(n_dim):
            assert input_data.get_component(point, dim) == basic_components[point][dim]

    data_array = input_data.to_components_array()
    assert isinstance(data_array, np.ndarray)
    assert data_array.shape == (n_points, n_dim)

    file_regression.check(f"{input_data!r}\n{input_data!s}")

    indices = range(0, n_points, 2)
    subset = input_data.get_subset(indices)
    assert subset.to_components_array().shape == (len(indices), n_dim)

    assert input_data.meta["access_components"]
    input_data.meta = {"access_components": False}
    assert not input_data.meta["access_components"]


class TestExtComponentsMemoryview:

    _tested = _types.InputDataExtComponentsMemoryview

    def test_by_parts(self):

        data = [
            np.array([[0, 0], [1, 1]]),
            np.array([[2, 2], [3, 3]])
        ]

        data_args, data_kwargs = recipes.prepare_components_array_from_parts(data)
        input_data = self._tested(*data_args, **data_kwargs)
        assert input_data.n_points == 4

        parts = list(input_data.by_parts())
        assert len(parts) == 2

        for i, part in enumerate(parts):
            np.testing.assert_array_equal(part, data[i])

        del input_data.meta["edges"]

        parts = list(input_data.by_parts())
        assert len(parts) == 1
        assert len(parts[0]) == 4


@pytest.mark.parametrize(
    "input_data_type,hook,kwargs",
    [
        (
            _types.InputDataExtDistancesLinearMemoryview,
            recipes.prepare_square_array_to_linear, {}
        ),
        (
            _types.InputDataExtDistancesMemoryview, recipes.prepare_to_array,
            {"order": "c", "dtype": P_AVALUE}
        ),
    ],
)
def test_input_data_init_distances(
        input_data_type,
        hook, kwargs,
        file_regression,
        basic_distances):

    n_points = len(basic_distances)

    if hook is None:
        hook = recipes.prepare_pass

    if kwargs is None:
        kwargs = {}

    data_args, data_kwargs = hook(basic_distances, **kwargs)
    meta = data_kwargs.get("meta", {})
    meta["dummy"] = "dummy"
    data_kwargs["meta"] = meta
    input_data = input_data_type(*data_args, **data_kwargs)

    assert input_data.meta["dummy"] == "dummy"
    assert input_data.data is not None
    assert input_data.n_points == n_points

    for a in range(n_points):
        for b in range(n_points):
            assert input_data.get_distance(a, b) == basic_distances[a][b]

    file_regression.check(f"{input_data!r}\n{input_data!s}")

    indices = range(0, n_points, 2)
    subset = input_data.get_subset(indices)
    assert subset.to_distance_array().shape == (len(indices), len(indices))

    assert input_data.meta["access_distances"]
    input_data.meta = {"access_distances": False}
    assert not input_data.meta["access_distances"]


@pytest.mark.parametrize(
    "input_data_type,hook,kwargs",
    [
        (
            _types.InputDataNeighbourhoodsSequence,
            None, None
        ),
        (
            _types.InputDataExtNeighbourhoodsMemoryview,
            recipes.prepare_padded_neighbourhoods_array, None
        ),
        (
            _types.InputDataExtNeighbourhoodsVector,
            None, None
        )
    ],
)
def test_input_data_init_neighbourhoods(
        input_data_type,
        hook, kwargs,
        file_regression,
        basic_neighbourhoods):

    n_points = len(basic_neighbourhoods)

    if hook is None:
        hook = recipes.prepare_pass

    if kwargs is None:
        kwargs = {}

    data_args, data_kwargs = hook(basic_neighbourhoods, **kwargs)
    meta = data_kwargs.get("meta", {})
    meta["dummy"] = "dummy"
    data_kwargs["meta"] = meta
    input_data = input_data_type(*data_args, **data_kwargs)

    assert input_data.meta["dummy"] == "dummy"
    assert input_data.data is not None
    assert input_data.n_neighbours is not None
    assert input_data.n_points == n_points

    n_neighbours_array = input_data.to_n_neighbours_array()
    for a in range(n_points):
        assert input_data.get_n_neighbours(a) == n_neighbours_array[a]
        for i in range(input_data.get_n_neighbours(a)):
            assert input_data.get_neighbour(a, i) == basic_neighbourhoods[a][i]

    file_regression.check(f"{input_data!r}\n{input_data!s}")

    indices = range(0, n_points, 2)
    subset = input_data.get_subset(indices)
    assert subset.to_neighbourhoods_array().shape[0] == len(indices)

    assert input_data.meta["access_neighbours"]
    input_data.meta = {"access_neighbours": False}
    assert not input_data.meta["access_neighbours"]


@pytest.mark.parametrize(
    "input_data_type,hook,kwargs",
    [
        (
            _types.InputDataSklearnKDTree, recipes.prepare_to_array,
            {"order": "c", "dtype": P_AVALUE}
        )
    ],
)
def test_input_data_trigger_neighbourhoods_computer(
        input_data_type,
        hook, kwargs,
        basic_components,
        basic_neighbourhoods):

    n_points = len(basic_components)

    if hook is None:
        hook = recipes.prepare_pass

    if kwargs is None:
        kwargs = {}

    data_args, data_kwargs = hook(basic_components, **kwargs)
    input_data = input_data_type(*data_args, **data_kwargs)
    input_data.compute_neighbourhoods(input_data, radius=1.1, is_sorted=True)
    assert input_data.n_neighbours is not None

    for a in range(n_points):
        for i in range(input_data.get_n_neighbours(a)):
            assert input_data.get_neighbour(a, i) == basic_neighbourhoods[a][i]

    input_data.compute_neighbourhoods(input_data, radius=1.1, is_selfcounting=False)
    for a in range(n_points):
        assert input_data.get_n_neighbours(a) == len(basic_neighbourhoods[a]) - 1
