import pytest
import numpy as np

from commonnn import _types


@pytest.mark.parametrize(
    "input_data_type,data,n_points,n_dim,queries",
    [
        (
            _types.InputDataComponentsSequence, [[0, 1], [1, 1], [2, 2]],
            3, 2, [(0, 0, 0), (0, 1, 1), (2, 0, 2)]
        )
        #    InputDataExtComponentsMemoryview,
        #    np.array([[0, 1], [0, 1]], order="C", dtype=P_AVALUE),
        #    2, 2, (0, 0), [(0, 0, 0)], [(1, 1, 1)]
        #),
    ],
)
def test_input_data_init_components(
        input_data_type,
        data,
        n_points, n_dim,
        queries):

    input_data = input_data_type(data)
    assert input_data.data is not None

    assert input_data.n_points == n_points
    assert input_data.n_dim == n_dim

    for point, dim, expected in queries:
        assert expected == input_data.get_component(point, dim)

    data_array = input_data.to_components_array()
    assert isinstance(data_array, np.ndarray)
    assert data_array.shape == (n_points, n_dim)

    subset = input_data.get_subset([1, 2])
    assert subset.to_components_array().shape == (2, n_dim)
