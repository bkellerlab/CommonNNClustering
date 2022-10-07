import pytest

from commonnn import _types


@pytest.mark.parametrize(
    "checker_type",
    [
        (_types.SimilarityCheckerContains),
        (_types.SimilarityCheckerSwitchContains),
        (_types.SimilarityCheckerExtContains),
        (_types.SimilarityCheckerExtSwitchContains),
        (_types.SimilarityCheckerExtScreensorted)
    ]
)
def test_init_checker(checker_type, file_regression):
    checker = checker_type()
    file_regression.check(f"{checker!r}\n{checker!s}")


@pytest.mark.parametrize(
    "checker_type,checker_is_ext,needs_sorted",
    [
        (_types.SimilarityCheckerContains, False, False),
        (_types.SimilarityCheckerSwitchContains, False, False),
        (_types.SimilarityCheckerExtContains, True, False),
        (_types.SimilarityCheckerExtSwitchContains, True, False),
        (_types.SimilarityCheckerExtScreensorted, True, True)
    ]
)
@pytest.mark.parametrize(
    "neighbours_type,args,kwargs,neighbours_is_ext,allows_sorted",
    [
        (_types.NeighboursList, (), {}, False, True),
        (_types.NeighboursSet, (), {}, False, False),
        (_types.NeighboursExtVector, (), {"initial_size": 10}, True, True),
        (_types.NeighboursExtVectorUnorderedSet, (), {"initial_size": 10}, True, True),
        (_types.NeighboursExtSet, (), {}, True, True),
        (_types.NeighboursExtUnorderedSet, (), {}, True, False),
    ],
)
@pytest.mark.parametrize(
    "members_a,members_b,c,is_sorted,expected",
    [
        ([], [], 0, True, True),
        ([], [], 1, True, False),
        ([1], [1], 2, True, False),
        ([1, 2, 3], [2, 5], 1, True, True),
        ([1, 2, 3], [2, 5, 8, 9], 2, True, False),
        ([1, 2, 3, 9, 10], [2, 5, 8, 9, 10], 3, True, True),
        ([3, 2, 1, 18, 9, 10], [2, 5, 9, 8, 10, 11, 18], 3, False, True),
    ]
)
def test_check(
        checker_type, checker_is_ext, needs_sorted,
        neighbours_type, args, kwargs, neighbours_is_ext,
        members_a, members_b, c, is_sorted, allows_sorted, expected):

    if checker_is_ext and (not neighbours_is_ext):
        # pytest.skip("Bad combination of component types.")
        return

    is_sorted *= allows_sorted

    if not is_sorted == needs_sorted:
        return

    neighbours_a = neighbours_type(*args, **kwargs)
    neighbours_b = neighbours_type(*args, **kwargs)
    for member in members_a:
        neighbours_a.assign(member)
    for member in members_b:
        neighbours_b.assign(member)

    cluster_params = _types.CommonNNParameters.from_mapping({
        "radius_cutoff": 0.0, "similarity_cutoff": c,
    })

    checker = checker_type()
    passed = checker.check(neighbours_a, neighbours_b, cluster_params)
    passed_swapped = checker.check(neighbours_b, neighbours_a, cluster_params)
    assert passed == expected == passed_swapped

    ns = checker.get(neighbours_a, neighbours_b, cluster_params)
    ns_swapped = checker.get(neighbours_b, neighbours_a, cluster_params)
    assert ns == len(set(members_a) & set(members_b)) == ns_swapped
