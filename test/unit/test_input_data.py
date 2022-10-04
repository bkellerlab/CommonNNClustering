import pytest
import numpy as np

from commonnn._primitive_types import P_AVALUE
from commonnn import _types


def to_array(x, **kwargs): return np.asarray(x, **kwargs)


def square_array_to_linear(x, **kwargs):
    indices = np.triu_indices_from(x, **kwargs)
    x = x[indices]
    return x


def pad_list(x):
    lengths = np.array([len(a) for a in x])
    pad_to = lengths.max()

    padded = []
    for i, a in enumerate(x):
        padded.append(a + [0] * (pad_to - lengths[i]))

    return np.asarray(padded)


@pytest.mark.parametrize(
    "input_data_type,hooks,kwargs",
    [
        (_types.InputDataComponentsSequence, None, None),
        (
            _types.InputDataExtComponentsMemoryview, [to_array],
            [{"order": "c", "dtype": P_AVALUE}]
        ),
        (_types.InputDataSklearnKDTree, [to_array], [{"order": "c", "dtype": P_AVALUE}])
    ],
)
def test_input_data_init_components(
        input_data_type,
        hooks, kwargs,
        file_regression,
        basic_components):

    n_points = len(basic_components)
    n_dim = len(basic_components[0])

    if hooks is not None:
        processed_components = basic_components
        for i, hook in enumerate(hooks):
            processed_components = hook(processed_components, **kwargs[i])
        input_data = input_data_type(processed_components)
    else:
        input_data = input_data_type(basic_components)

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


@pytest.mark.parametrize(
    "input_data_type,hooks,kwargs",
    [
        (
            _types.InputDataExtDistancesLinearMemoryview,
            [to_array, square_array_to_linear], [{}, {"k": 1}]
        )
    ],
)
def test_input_data_init_distances(
        input_data_type,
        hooks, kwargs,
        file_regression,
        basic_distances):

    n_points = len(basic_distances)

    if hooks is not None:
        processed_distances = basic_distances
        for i, hook in enumerate(hooks):
            processed_distances = hook(processed_distances, **kwargs[i])
        input_data = input_data_type(processed_distances)
    else:
        input_data = input_data_type(basic_distances)

    assert input_data.data is not None
    assert input_data.n_points == n_points

    for a in range(n_points):
        for b in range(n_points):
            assert input_data.get_distance(a, b) == basic_distances[a][b]

    file_regression.check(f"{input_data!r}\n{input_data!s}")


@pytest.mark.parametrize(
    "input_data_type,hooks,kwargs",
    [
        (
            _types.InputDataNeighbourhoodsSequence,
            [pad_list], [{}]
        )
    ],
)
def test_input_data_init_neighbourhoods(
        input_data_type,
        hooks, kwargs,
        file_regression,
        basic_neighbourhoods):

    n_points = len(basic_neighbourhoods)

    if hooks is not None:
        processed_neighbourhoods = basic_neighbourhoods
        for i, hook in enumerate(hooks):
            processed_neighbourhoods = hook(processed_neighbourhoods, **kwargs[i])
        input_data = input_data_type(processed_neighbourhoods)
    else:
        input_data = input_data_type(basic_neighbourhoods)

    assert input_data.data is not None
    assert input_data.n_points == n_points

    for a in range(n_points):
        for i in range(input_data.get_n_neighbours(a)):
            assert input_data.get_neighbour(a, i) == basic_neighbourhoods[a][i]

    file_regression.check(f"{input_data!r}\n{input_data!s}")
