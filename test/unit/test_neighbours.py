from operator import ne
import pytest

from commonnn import _types


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

    neighbours.assign(5)
    assert neighbours.n_points == n_points + 1
    if ordered:
        expected = list(args[0]) + [5]
        for i in range(neighbours.n_points):
            assert neighbours.get_member(i) == expected[i]

    assert neighbours.contains(5)
    assert not neighbours.contains(99)
    assert neighbours.enough(3)
    assert not neighbours.enough(4)

    neighbours.reset()
    assert neighbours.n_points == 0 == neighbours._n_points
