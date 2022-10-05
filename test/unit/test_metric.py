import math

import numpy as np
import pytest

from commonnn._primitive_types import P_AVALUE
from commonnn import _types


def ref_distance_euclidean(a, b):
    total = 0
    for component_a, component_b in zip(a, b):
        total += (component_a - component_b) ** 2
    return math.sqrt(total)


def test_ref_distance_euclidean():
    assert ref_distance_euclidean((0, 0), (0, 0)) == 0
    assert ref_distance_euclidean((0, 0), (0, 1)) == 1
    assert ref_distance_euclidean((0, -1), (0, 1)) == 2
    assert ref_distance_euclidean((1, 1), (1, 1)) == 0


@pytest.mark.parametrize(
    "metric,metric_args,metric_kwargs,isinstance_of",
    [
        (_types.MetricDummy, (), {}, [_types.Metric]),
        (_types.MetricPrecomputed, (), {}, [_types.Metric]),
        (_types.MetricEuclidean, (), {}, [_types.Metric]),
        (_types.MetricEuclideanReduced, (), {}, [_types.Metric]),
        (_types.MetricExtDummy, (), {}, [_types.Metric, _types.MetricExtInterface]),
        (_types.MetricExtPrecomputed, (), {}, [_types.Metric, _types.MetricExtInterface]),
        (_types.MetricExtEuclidean, (), {}, [_types.Metric, _types.MetricExtInterface]),
        (_types.MetricExtEuclideanReduced, (), {}, [_types.Metric, _types.MetricExtInterface]),
        (
            _types.MetricExtEuclideanPeriodicReduced,
            (np.ones(2),), {},
            [_types.Metric, _types.MetricExtInterface]
        ),
    ]
)
def test_inheritance(metric, metric_args, metric_kwargs, isinstance_of):
    _metric = metric(*metric_args, **metric_kwargs)
    assert all([isinstance(_metric, x) for x in isinstance_of])


@pytest.mark.parametrize(
    "metric,metric_args,metric_kwargs,metric_is_ext,ref_func",
    [
        (
            _types.MetricEuclidean, (), {}, False,
            ref_distance_euclidean,
        ),
        (
            _types.MetricEuclideanReduced, (), {}, False,
            ref_distance_euclidean,
        ),
        (
            _types.MetricExtEuclidean, (), {}, True,
            ref_distance_euclidean,
        ),
        (
            _types.MetricExtEuclideanReduced, (), {}, True,
            ref_distance_euclidean,
        ),
    ]
)
@pytest.mark.parametrize(
    "input_data_type,data,other_data,input_is_ext",
    [
        (
            _types.InputDataExtComponentsMemoryview,
            np.array([
                [0, 0, 0],
                [1, 1, 1],
                [1.2, 1.5, 1.3],
            ], order="C", dtype=P_AVALUE
            ),
            np.array([
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [1.2, 1.5, 1.3],
                [4.3, 2.5, 0.7],
            ], order="C", dtype=P_AVALUE
            ),
            True
        ),
    ],
)
def test_calc_distance(
        metric, metric_args, metric_kwargs, metric_is_ext,
        input_data_type, data, other_data, input_is_ext, ref_func):

    if metric_is_ext and (not input_is_ext):
        # pytest.skip("Bad combination of component types.")
        return

    _metric = metric(*metric_args, **metric_kwargs)

    input_data = input_data_type(data)

    for i in range(input_data.n_points):
        for j in range(i + 1, input_data.n_points):
            a, b = zip(
                *(
                    (
                        input_data.get_component(i, d),
                        input_data.get_component(j, d),
                    )
                    for d in range(input_data.n_dim)
                )
            )
            ref_distance = _metric.adjust_radius(ref_func(a, b))

            distance = _metric.calc_distance(i, j, input_data)

            np.testing.assert_approx_equal(
                distance, ref_distance, significant=12
            )

    other_input_data = input_data_type(other_data)

    for i in range(other_input_data.n_points):
        for j in range(input_data.n_points):
            a, b = zip(
                *(
                    (
                        other_input_data.get_component(i, d),
                        input_data.get_component(j, d),
                    )
                    for d in range(input_data.n_dim)
                )
            )
            ref_distance = _metric.adjust_radius(ref_func(a, b))

            distance = _metric.calc_distance_other(
                i, j, input_data, other_input_data
            )

            np.testing.assert_approx_equal(
                distance, ref_distance, significant=12
            )