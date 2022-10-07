import pytest

from commonnn import _fit, _types


@pytest.mark.parametrize(
    "fitter_type,args",
    [
        (_fit.FitterCommonNNBFS, (None, None, None, None, None)),
        (_fit.FitterExtCommonNNBFS, (None, None, None, None, None)),
    ]
)
def test_init_fitter(fitter_type, args, file_regression):
    fitter = fitter_type(*args)
    if hasattr(fitter, "get_fit_signature"):
        signature = "\n" + fitter.get_fit_signature()
    else:
        signature = ""

    file_regression.check(f"{fitter!r}\n{fitter!s}{signature}")


@pytest.mark.parametrize(
    "fitter_type,args,expected",
    [
        (
            _fit.FitterCommonNNBFS, (None, None, None, None, None), {
                "radius_cutoff": 2.,
                "similarity_cutoff": 2,
                "_support_cutoff": 2,
                "start_label": 1
            }
        ),
        (
            _fit.FitterExtCommonNNBFS, (None, None, None, None, None), {
                "radius_cutoff": 2.,
                "similarity_cutoff": 2,
                "_support_cutoff": 2,
                "start_label": 1
            }
        ),
        (
            _fit.FitterExtCommonNNBFS, (
                _types.NeighboursGetterExtBruteForce(
                    distance_getter=_types.DistanceGetterExtMetric(
                        metric=_types.MetricExtEuclideanReduced()
                    )
                ), None, None, None, None
            ), {
                "radius_cutoff": 4.,
                "similarity_cutoff": 4,
                "_support_cutoff": 3,
                "start_label": 1
            }
        )
    ]
)
def test_make_parameters(fitter_type, args, expected):
    parameters = {
        "radius_cutoff": 2.,
        "similarity_cutoff": 2,
    }

    fitter = fitter_type(*args)
    cluster_params = fitter.make_parameters(**parameters)

    assert isinstance(cluster_params, fitter._parameter_type)
    assert expected == cluster_params.to_dict()
