import random

import numpy as np
import pytest

from commonnn import _types


@pytest.mark.parametrize(
    "neighbours_type",
    [
        (_types.NeighboursList),
        (_types.NeighboursSet),
        (_types.NeighboursExtVector),
        (_types.NeighboursExtVectorUnorderedSet),
        (_types.NeighboursExtSet),
        (_types.NeighboursExtUnorderedSet)
    ]
)
def test_init_neighbours(neighbours_type, file_regression):
    neighbours = neighbours_type()
    file_regression.check(f"{neighbours!r}\n{neighbours!s}")


@pytest.mark.parametrize(
    "neighbours_type,args,kwargs,n_points,ordered",
    [
        (_types.NeighboursList, ([0, 1, 3],), {}, 3, True),
        (_types.NeighboursSet, ({0, 1, 3},), {}, 3, True),
        (_types.NeighboursExtVector, ([0, 1, 3],), {"initial_size": 5}, 3, True),
        (_types.NeighboursExtVectorUnorderedSet, ([0, 1, 3],), {"initial_size": 5}, 3, True),
        (_types.NeighboursExtSet, ([0, 1, 3],), {}, 3, True),
        (_types.NeighboursExtUnorderedSet, ([0, 1, 3],), {}, 3, False)
    ]
)
def test_neighbours(neighbours_type, args, kwargs, n_points, ordered):
    neighbours = neighbours_type(*args, **kwargs)
    assert neighbours.n_points == n_points == neighbours._n_points
    if hasattr(neighbours, "neighbours"):
        assert neighbours.neighbours is not None

    if ordered:
        np.testing.assert_array_equal(
            neighbours.to_neighbours_array(),
            np.array(list(args[0]))
        )
    else:
        assert set(list(neighbours.to_neighbours_array())) == set(args[0])

    neighbours.assign(5)
    assert neighbours.n_points == n_points + 1
    if ordered:
        expected = list(args[0]) + [5]
        for i in range(neighbours.n_points):
            assert neighbours.get_member(i) == expected[i]

    expected = set(list(args[0]) + [5])
    members = list(range(n_points))
    random.shuffle(members)
    for i in members:
        assert neighbours.get_member(i) in expected

    assert neighbours.contains(5)
    assert not neighbours.contains(99)
    assert neighbours.enough(3)
    assert not neighbours.enough(4)

    neighbours.reset()
    assert neighbours.n_points == 0 == neighbours._n_points
